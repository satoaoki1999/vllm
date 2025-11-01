import itertools
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from typing import (Any, ClassVar, Dict, List, Optional, Sequence, Tuple,
                    Union, cast, overload)

from tqdm import tqdm

from vllm.engine.arg_utils import EngineArgs
from vllm.engine.llm_engine import LLMEngine
from vllm.entrypoints.chat_utils import (ChatCompletionMessageParam,
                                         apply_hf_chat_template,
                                         apply_mistral_chat_template,
                                         parse_chat_messages)
from vllm.inputs import PromptType, TextPrompt, TokensPrompt
from vllm.inputs.parse import parse_and_batch_prompt
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.model_executor.guided_decoding.guided_fields import (
    GuidedDecodingRequest, LLMGuidedOptions)
from vllm.outputs import EmbeddingRequestOutput, RequestOutput
from vllm.pooling_params import PoolingParams
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.sampling_params import (BeamSearchParams, GuidedDecodingParams,
                                  RequestOutputKind, SamplingParams)
from vllm.transformers_utils.tokenizer import (AnyTokenizer, MistralTokenizer,
                                               get_cached_tokenizer)
from vllm.transformers_utils.tokenizer_group import TokenizerGroup
from vllm.usage.usage_lib import UsageContext
from vllm.utils import Counter, deprecate_kwargs, is_list_of

import torch
import time

logger = init_logger(__name__)


@dataclass
class BeamSearchSequence:
    """A sequence for beam search.
    It keeps track of the tokens and the log probability of the sequence.
    The text field is optional and will only be filled when the sequence is
    about to be returned to the user.
    """
    # The tokens includes the prompt.
    tokens: List[int]
    cum_logprob: float = 0.0
    text: Optional[str] = None


@dataclass
class BeamSearchOutput:
    """The output of beam search.
    It contains the list of the best beam search sequences.
    The length of the list is equal to the beam width.
    """
    sequences: List[BeamSearchSequence]


class BeamSearchInstance:

    def __init__(self, prompt_tokens: List[int]):
        self.beams: List[BeamSearchSequence] = [
            BeamSearchSequence(tokens=prompt_tokens)
        ]
        self.completed: List[BeamSearchSequence] = []


class LLM:
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a batch of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management.

    Args:
        model: The name or path of a HuggingFace Transformers model.
        tokenizer: The name or path of a HuggingFace Transformers tokenizer.
        tokenizer_mode: The tokenizer mode. "auto" will use the fast tokenizer
            if available, and "slow" will always use the slow tokenizer.
        skip_tokenizer_init: If true, skip initialization of tokenizer and
            detokenizer. Expect valid prompt_token_ids and None for prompt
            from the input.
        trust_remote_code: Trust remote code (e.g., from HuggingFace) when
            downloading the model and tokenizer.
        tensor_parallel_size: The number of GPUs to use for distributed
            execution with tensor parallelism.
        dtype: The data type for the model weights and activations. Currently,
            we support `float32`, `float16`, and `bfloat16`. If `auto`, we use
            the `torch_dtype` attribute specified in the model config file.
            However, if the `torch_dtype` in the config is `float32`, we will
            use `float16` instead.
        quantization: The method used to quantize the model weights. Currently,
            we support "awq", "gptq", and "fp8" (experimental).
            If None, we first check the `quantization_config` attribute in the
            model config file. If that is None, we assume the model weights are
            not quantized and use `dtype` to determine the data type of
            the weights.
        revision: The specific model version to use. It can be a branch name,
            a tag name, or a commit id.
        tokenizer_revision: The specific tokenizer version to use. It can be a
            branch name, a tag name, or a commit id.
        seed: The seed to initialize the random number generator for sampling.
        gpu_memory_utilization: The ratio (between 0 and 1) of GPU memory to
            reserve for the model weights, activations, and KV cache. Higher
            values will increase the KV cache size and thus improve the model's
            throughput. However, if the value is too high, it may cause out-of-
            memory (OOM) errors.
        swap_space: The size (GiB) of CPU memory per GPU to use as swap space.
            This can be used for temporarily storing the states of the requests
            when their `best_of` sampling parameters are larger than 1. If all
            requests will have `best_of=1`, you can safely set this to 0.
            Otherwise, too small values may cause out-of-memory (OOM) errors.
        cpu_offload_gb: The size (GiB) of CPU memory to use for offloading
            the model weights. This virtually increases the GPU memory space
            you can use to hold the model weights, at the cost of CPU-GPU data
            transfer for every forward pass.
        enforce_eager: Whether to enforce eager execution. If True, we will
            disable CUDA graph and always execute the model in eager mode.
            If False, we will use CUDA graph and eager execution in hybrid.
        max_context_len_to_capture: Maximum context len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode (DEPRECATED. Use `max_seq_len_to_capture` instead).
        max_seq_len_to_capture: Maximum sequence len covered by CUDA graphs.
            When a sequence has context length larger than this, we fall back
            to eager mode. Additionally for encoder-decoder models, if the
            sequence length of the encoder input is larger than this, we fall
            back to the eager mode.
        disable_custom_all_reduce: See ParallelConfig
        **kwargs: Arguments for :class:`~vllm.EngineArgs`. (See
            :ref:`engine_args`)

    Note:
        This class is intended to be used for offline inference. For online
        serving, use the :class:`~vllm.AsyncLLMEngine` class instead.
    """

    DEPRECATE_LEGACY: ClassVar[bool] = False
    """A flag to toggle whether to deprecate the legacy generate/encode API."""

    @classmethod
    @contextmanager
    def deprecate_legacy_api(cls):
        cls.DEPRECATE_LEGACY = True

        yield

        cls.DEPRECATE_LEGACY = False

    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_context_len_to_capture: Optional[int] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:
        '''
        LLM constructor.

        Note: if enforce_eager is unset (enforce_eager is None)
        it defaults to False.
        '''

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        removed_vision_keys = (
            "image_token_id",
            "image_feature_size",
            "image_input_shape",
            "image_input_type",
        )
        if any(k in kwargs for k in removed_vision_keys):
            raise TypeError(
                "There is no need to pass vision-related arguments anymore.")
        engine_args = EngineArgs(
            model=model,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,
            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_context_len_to_capture=max_context_len_to_capture,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            mm_processor_kwargs=mm_processor_kwargs,
            **kwargs,
        )
        self.llm_engine = LLMEngine.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)
        self.request_counter = Counter()

    def get_tokenizer(self) -> AnyTokenizer:
        return self.llm_engine.get_tokenizer_group(TokenizerGroup).tokenizer

    def set_tokenizer(self, tokenizer: AnyTokenizer) -> None:
        tokenizer_group = self.llm_engine.get_tokenizer_group(TokenizerGroup)

        # While CachedTokenizer is dynamic, have no choice but
        # compare class name. Misjudgment will arise from
        # user-defined tokenizer started with 'Cached'
        if tokenizer.__class__.__name__.startswith("Cached"):
            tokenizer_group.tokenizer = tokenizer
        else:
            tokenizer_group.tokenizer = get_cached_tokenizer(tokenizer)

    @overload  # LEGACY: single (prompt + optional token ids)
    def generate(
        self,
        prompts: str,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    def generate(
        self,
        prompts: List[str],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    def generate(
        self,
        prompts: Optional[str] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        *,
        prompt_token_ids: List[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    def generate(
        self,
        prompts: Optional[List[str]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        *,
        prompt_token_ids: List[List[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    def generate(
        self,
        prompts: None,
        sampling_params: None,
        prompt_token_ids: Union[List[int], List[List[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @overload
    def generate(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        *,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[RequestOutput]:
        ...

    @deprecate_kwargs(
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'prompts' parameter instead.",
    )
    def generate(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, List[str]]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        Sequence[SamplingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        guided_options_request: Optional[Union[LLMGuidedOptions,
                                               GuidedDecodingRequest]] = None,
        priority: Optional[List[int]] = None,
    ) -> List[RequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            sampling_params: The sampling parameters for text generation. If
                None, we use the default sampling parameters.
                When it is a single value, it is applied to every prompt.
                When it is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.
            priority: The priority of the requests, if any.
                Only applicable when priority scheduling policy is enabled.

        Returns:
            A list of ``RequestOutput`` objects containing the
            generated completions in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        if self.llm_engine.model_config.embedding_mode:
            raise ValueError(
                "LLM.generate() is only supported for (conditional) generation "
                "models (XForCausalLM, XForConditionalGeneration).")

        if prompt_token_ids is not None:
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if isinstance(guided_options_request, dict):
            if len(guided_options_request) > 1:
                raise ValueError(
                    "You can only use one guided decoding but multiple is "
                    f"specified: {guided_options_request}")
            guided_options_request = GuidedDecodingRequest(
                **guided_options_request)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=sampling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            guided_options=guided_options_request,
            priority=priority)

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, RequestOutput)

    def beam_search(
        self,
        prompts: List[Union[str, List[int]]],
        params: BeamSearchParams,
    ) -> List[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.

        TODO: how does beam search work together with length penalty, frequency
        penalty, and stopping criteria, etc.?
        """

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos

        tokenizer = self.get_tokenizer()
        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        beam_search_params = SamplingParams(logprobs=2 * beam_width,
                                            max_tokens=1,
                                            temperature=temperature)
        instances: List[BeamSearchInstance] = []

        for prompt in prompts:
            prompt_tokens = prompt if isinstance(
                prompt, list) else tokenizer.encode(prompt)
            instances.append(BeamSearchInstance(prompt_tokens))

        for _ in range(max_tokens):
            all_beams: List[BeamSearchSequence] = list(
                sum((instance.beams for instance in instances), []))
            pos = [0] + list(
                itertools.accumulate(
                    len(instance.beams) for instance in instances))
            instance_start_and_end: List[Tuple[int, int]] = list(
                zip(pos[:-1], pos[1:]))

            if len(all_beams) == 0:
                break

            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens)
                for beam in all_beams
            ]

            # only runs for one step
            # we don't need to use tqdm here
            output = self.generate(prompts_batch,
                                   sampling_params=beam_search_params,
                                   use_tqdm=False)

            for (start, end), instance in zip(instance_start_and_end,
                                              instances):
                instance_new_beams = []
                for i in range(start, end):
                    current_beam = all_beams[i]
                    result = output[i]

                    if result.outputs[0].logprobs is not None:
                        # if `result.outputs[0].logprobs` is None, it means
                        # the sequence is completed because of the max-model-len
                        # or abortion. we don't need to add it to the new beams.
                        logprobs = result.outputs[0].logprobs[0]
                        for token_id, logprob_obj in logprobs.items():
                            new_beam = BeamSearchSequence(
                                tokens=current_beam.tokens + [token_id],
                                cum_logprob=current_beam.cum_logprob +
                                logprob_obj.logprob)

                            if token_id == tokenizer.eos_token_id and \
                                    not ignore_eos:
                                instance.completed.append(new_beam)
                            else:
                                instance_new_beams.append(new_beam)
                sorted_beams = sorted(instance_new_beams,
                                      key=lambda x: x.cum_logprob,
                                      reverse=True)
                instance.beams = sorted_beams[:beam_width]

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(instance.completed,
                                      key=lambda x: x.cum_logprob,
                                      reverse=True)
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def beam_search_by_official_v063(
        self,
        prompts: List[Union[str, List[int]]],
        params: BeamSearchParams,
    ) -> List[BeamSearchOutput]:
        return self.beam_search(prompts, params)

    def beam_search_by_official_v063_opt(
        self,
        prompts: List[Union[str, List[int]]],
        params: BeamSearchParams,
    ) -> List[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.

        TODO: how does beam search work together with length penalty, frequency
        penalty, and stopping criteria, etc.?
        """

        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos

        tokenizer = self.get_tokenizer()
        # generate 2 * beam_width candidates at each step
        # following the huggingface transformers implementation
        # at https://github.com/huggingface/transformers/blob/e15687fffe5c9d20598a19aeab721ae0a7580f8a/src/transformers/generation/beam_search.py#L534 # noqa
        beam_search_params = SamplingParams(logprobs=2 * beam_width,
                                            max_tokens=1,
                                            temperature=temperature)
        instances: List[BeamSearchInstance] = []

        for prompt in prompts:
            prompt_tokens = prompt if isinstance(
                prompt, list) else tokenizer.encode(prompt)
            instances.append(BeamSearchInstance(prompt_tokens))

        for _ in range(max_tokens):
            all_beams: List[BeamSearchSequence] = list(
                sum((instance.beams for instance in instances), []))
            pos = [0] + list(
                itertools.accumulate(
                    len(instance.beams) for instance in instances))
            instance_start_and_end: List[Tuple[int, int]] = list(
                zip(pos[:-1], pos[1:]))

            if len(all_beams) == 0:
                break

            prompts_batch = [
                TokensPrompt(prompt_token_ids=beam.tokens)
                for beam in all_beams
            ]

            # only runs for one step
            # we don't need to use tqdm here
            output = self.generate(prompts_batch,
                                   sampling_params=beam_search_params,
                                   use_tqdm=False)

            for (start, end), instance in zip(instance_start_and_end,
                                              instances):
                instance_new_beams = []
                for i in range(start, end):
                    current_beam = all_beams[i]
                    result = output[i]

                    if result.outputs[0].logprobs is not None:
                        # if `result.outputs[0].logprobs` is None, it means
                        # the sequence is completed because of the max-model-len
                        # or abortion. we don't need to add it to the new beams.
                        logprobs = result.outputs[0].logprobs[0]
                        for token_id, logprob_obj in logprobs.items():
                            new_beam = BeamSearchSequence(
                                tokens=current_beam.tokens + [token_id],
                                cum_logprob=current_beam.cum_logprob +
                                logprob_obj.logprob)

                            if token_id == tokenizer.eos_token_id and \
                                    not ignore_eos:
                                instance.completed.append(new_beam)
                            else:
                                instance_new_beams.append(new_beam)
                sorted_beams = sorted(instance_new_beams,
                                      key=lambda x: x.cum_logprob,
                                      reverse=True)
                instance.beams = sorted_beams[:beam_width]
                if len(instance.completed) >= beam_width and sorted_beams[0].cum_logprob < min([completed_seq.cum_logprob for completed_seq in instance.completed]):
                    instance.beams.clear()

        outputs = []
        for instance in instances:
            instance.completed.extend(instance.beams)
            sorted_completed = sorted(instance.completed,
                                      key=lambda x: x.cum_logprob,
                                      reverse=True)
            best_beams = sorted_completed[:beam_width]

            for beam in best_beams:
                beam.text = tokenizer.decode(beam.tokens)
            outputs.append(BeamSearchOutput(sequences=best_beams))

        return outputs

    def beam_search_by_transformers_vectoring(
        self,
        prompts: List[Union[str, List[int]]],
        params: BeamSearchParams,
    ) -> List[BeamSearchOutput]:
        """
        Generate sequences using beam search.

        Args:
            prompts: A list of prompts. Each prompt can be a string or a list
                of token IDs.
            params: The beam search parameters.
        """
        tokenizer = self.get_tokenizer()

        # step1: param process
        temperature = params.temperature
        ignore_eos = params.ignore_eos
        do_sample = params.do_sample
        early_stopping = params.early_stopping
        length_penalty = params.length_penalty
        max_length = params.max_tokens
        num_beams = params.beam_width
        num_return_sequences = params.num_return_sequences if params.num_return_sequences is not None else num_beams
        pad_token_id = params.pad_token_id if params.pad_token_id is not None else tokenizer.pad_token_id
        eos_token_id = params.eos_token_id if params.eos_token_id is not None else tokenizer.eos_token_id
        vocab_size = self.llm_engine.model_config.get_vocab_size()
        batch_size = len(prompts)

        # step2: encode
        if isinstance(prompts[0], str):  # prompts are strings
            encoded_inputs = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
            )
            input_ids = encoded_inputs["input_ids"]
        else:  # prompts are token ids
            max_len = max(len(seq) for seq in prompts)
            input_ids = torch.full(
                (batch_size, max_len), pad_token_id, dtype=torch.long)
            for i, seq in enumerate(prompts):
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        device = input_ids.device  # cpu
        _, cur_len = input_ids.shape

        # step3: vector based beam-search(from transformer)

        # expand to size [batch * num_beams, prompts_max_len, vocab_size]
        input_ids = input_ids.repeat_interleave(
            num_beams, dim=0)

        decoder_prompt_len = cur_len

        # At each beam search step, we want to keep top K [K = (number of EOS tokens + 1) * `num_beams`] candidates
        # with the highest log-probabilities, or sample K continuations without replacement. We gather the top K
        # (as opposed to `num_beams`, or any number lower than K) so that we have at least `num_beams` sequences
        # non-finished to continue the live beam search, in case the top `num_beams` all select an EOS token.
        n_eos_tokens = len(eos_token_id) if isinstance(
            eos_token_id, list) else 1
        beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
        top_num_beam_mask = torch.cat(
            (torch.ones((num_beams), dtype=torch.bool),
             torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)),
            dim=0
        ).to(device)

        # per batch, beam-item holding current token in loop and completed sequences
        output_fill_value = pad_token_id or eos_token_id[0] if eos_token_id is not None else -1
        running_sequences = torch.full(
            (batch_size, num_beams, max_length),
            fill_value=output_fill_value,
            dtype=torch.long,
            device=device,
        )
        running_sequences[:, :, :cur_len] = self._unflatten_beam_dim(
            input_ids, batch_size, num_beams)
        sequences = running_sequences.detach().clone()

        # per batch, beam-item score, logprobs
        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        running_beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device)
        running_beam_scores[:, 1:] = -1e9
        beam_scores = torch.full(
            (batch_size, num_beams), fill_value=-1e9, dtype=torch.float, device=device)

        # per batch, beam-item state bit indicating if sentence has finished.
        is_sent_finished = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=device)

        # per batch state bit indicating if there is a possibility to improve the best finished sentence.
        is_early_stop_heuristic_unsatisfied = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=device)

        # per batch, beam-item state bit indicating if there are valid continuations.
        next_token_hits_stopping_criteria = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=device)

        # per batch selected beam indices
        running_beam_indices = torch.full(
            (batch_size, num_beams, max_length - cur_len),
            fill_value=-1,
            dtype=torch.int32,
            device=device
        )
        beam_indices = running_beam_indices.detach().clone()

        while cur_len < max_length:
            # a. Forward current tokens, obtain the logits
            flat_running_sequences = self._flatten_beam_dim(
                running_sequences[:, :, :cur_len])

            model_inputs = self._get_token_prompts(flat_running_sequences)
            model_outputs = self.generate(prompts=model_inputs, sampling_params=SamplingParams(logprobs=beams_to_keep // num_beams,  # beams_to_keep // num_beams == 2
                                                                                               max_tokens=1,
                                                                                               temperature=temperature),
                                          use_tqdm=False)

            # b. Compute log probs -- get log probabilities from logits, process logits with processors (*e.g.*
            # `temperature`, ...), and add new logprobs to existing running logprobs scores.
            log_probs = torch.full(
                (batch_size * num_beams, vocab_size), -1e9, dtype=torch.bfloat16)
            for batch_idx in range(batch_size):
                for beam_idx in range(num_beams):
                    idx = batch_idx * num_beams + beam_idx
                    # for cand_idx, (token_id, log_prob_obj) in enumerate(model_outputs[idx].outputs[0].logprobs[0].items()):
                    kv = list(
                        model_outputs[idx].outputs[0].logprobs[0].items())
                    for cand_idx in range(num_beams):
                        (token_id, log_prob_obj) = kv[cand_idx]
                        log_probs[idx][token_id] = log_prob_obj.logprob
            log_probs = self._unflatten_beam_dim(
                log_probs, batch_size, num_beams)
            log_probs = log_probs + running_beam_scores[:, :, None]
            log_probs = torch.reshape(
                log_probs, (batch_size, num_beams * vocab_size))

            # c. Retrieve top-K continuations, i.e. select the next token (greedy or sampling) and then keep the best
            # continuations among all beams based on the accumulated scores.
            topk_log_probs, topk_running_sequences, topk_running_beam_indices = self._get_top_k_continuations(
                accumulated_log_probs=log_probs,
                running_sequences=running_sequences,
                running_beam_indices=running_beam_indices,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                do_sample=do_sample,
                beams_to_keep=beams_to_keep,
                num_beams=num_beams,
                vocab_size=vocab_size,
                batch_size=batch_size,
            )

            # d. Check which running sequences have finished
            if not ignore_eos:
                next_token_hits_stopping_criteria = topk_running_sequences[:,
                                                                           :, cur_len] == eos_token_id

            # e. Get the non-finished running `num_beams` sequences for the next generation step
            running_sequences, running_beam_scores, running_beam_indices = self._get_running_beams_for_next_iteration(
                topk_log_probs=topk_log_probs,
                topk_running_sequences=topk_running_sequences,
                topk_running_beam_indices=topk_running_beam_indices,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                num_beams=num_beams,
            )

            # f. Update the completed beams if a new high score in a finished sequence is found
            sequences, beam_scores, beam_indices, is_sent_finished = self._update_finished_beams(
                sequences=sequences,
                topk_running_sequences=topk_running_sequences,
                beam_scores=beam_scores,
                topk_log_probs=topk_log_probs,
                beam_indices=beam_indices,
                topk_running_beam_indices=topk_running_beam_indices,
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                is_sent_finished=is_sent_finished,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                top_num_beam_mask=top_num_beam_mask,
                num_beams=num_beams,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

            # g. Prepare remaining data for the next iteration, including computing the stopping condition for
            # beam search as a whole (as opposed to individual beams, i.e. `stopping_criteria`)
            cur_len += 1
            is_early_stop_heuristic_unsatisfied = self._check_early_stop_heuristic(
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                running_beam_scores=running_beam_scores,
                beam_scores=beam_scores,
                is_sent_finished=is_sent_finished,
                cur_len=cur_len,
                max_length=max_length,
                decoder_prompt_len=decoder_prompt_len,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
            )
            if not self._beam_search_has_unfinished_sequences(is_early_stop_heuristic_unsatisfied,
                                                              is_sent_finished,
                                                              next_token_hits_stopping_criteria,
                                                              early_stopping):
                break  # if all sequences are finished, break directly

        # step4：output
        sequences = self._flatten_beam_dim(
            sequences[:, :num_return_sequences, :])
        beam_scores = self._flatten_beam_dim(
            beam_scores[:, :num_return_sequences])

        outputs = []
        for i in range(batch_size):
            batch_outputs = []
            for j in range(num_return_sequences):
                idx = i * num_return_sequences + j
                seq = sequences[idx].tolist()
                if pad_token_id in seq:
                    seq = seq[:seq.index(pad_token_id)]
                if eos_token_id in seq:
                    eos_pos = seq.index(eos_token_id)
                    seq = seq[:eos_pos + 1]
                # seq.append(eos_token_id)

                score = beam_scores[idx].item() if idx < len(
                    beam_scores) else 0.0

                batch_outputs.append(BeamSearchSequence(
                    text=tokenizer.decode(seq),
                    tokens=seq,
                    cum_logprob=score
                ))
            outputs.append(BeamSearchOutput(sequences=batch_outputs))
        return outputs

    @staticmethod
    def _get_token_prompts(tensor: torch.Tensor):
        # tensor shape: batch * beam * ...
        tokens_id_list = tensor.tolist()
        return [TokensPrompt(prompt_token_ids=tokens_id) for tokens_id in tokens_id_list]

    @staticmethod
    def _flatten_beam_dim(tensor: torch.Tensor) -> torch.Tensor:
        """[batch_size, num_beams, ...] -> [batch_size * num_beams, ...]"""
        shape = list(tensor.shape)
        return torch.reshape(tensor, [shape[0] * shape[1]] + shape[2:])

    @staticmethod
    def _unflatten_beam_dim(tensor: torch.Tensor, batch_size: int, num_beams: int) -> torch.Tensor:
        """[batch_size * num_beams, ...] -> [batch_size, num_beams, ...]"""
        shape = list(tensor.shape)
        return torch.reshape(tensor, [batch_size, num_beams] + shape[1:])

    @staticmethod
    def _gather_beams(tensor: torch.Tensor, beam_indices: torch.Tensor) -> torch.Tensor:
        """Gathers the beam slices indexed by beam_indices into new beam array."""
        while len(beam_indices.shape) < len(tensor.shape):
            beam_indices = beam_indices.unsqueeze(-1)
        gathered_tensor = torch.take_along_dim(
            input=tensor, indices=beam_indices, dim=1)
        return gathered_tensor

    @staticmethod
    def _check_early_stop_heuristic(
        is_early_stop_heuristic_unsatisfied: torch.Tensor,
        running_beam_scores: torch.Tensor,
        beam_scores: torch.Tensor,
        is_sent_finished: torch.Tensor,
        cur_len: int,
        max_length: int,
        decoder_prompt_len: int,
        early_stopping: Union[bool, str],
        length_penalty: float,
    ):
        """
        Determine whether early stopping is possible by checking if the best possible score of running beams
        could still improve upon the finished ones.

        Mechanism:
        - Without a length penalty, beam scores typically decrease as more tokens are generated.
        So, if the *best possible* score from any running beam is already worse than the *worst* finished beam,
        we can safely stop early.
        - With a length penalty, scores may increase with longer sequences. In this case, we use heuristics
        to estimate the best possible score — though this estimate may not always be correct — and stop
        if no further improvement seems likely.

        We apply different heuristics depending on the value of `early_stopping`:
        1. `early_stopping == False`:
        -> Use a heuristic that assumes the best score comes from the current length minus the decoder prompt length.
        -> See detailed discussion: https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565

        2. `early_stopping == "never"`:
        -> Estimate the best score using either `max_length` or `cur_len`, depending on the sign of `length_penalty`.
        -> A positive length penalty favors longer sequences, so we use `max_length` in that case.

        NOTE: the canonical beam search implementation can be replicated with `early_stopping="never"` and
        `length_penalty=0.0`, which are NOT the default flags. The default behavior was empirically found to produce
        better sequences (prior to 2022), and changing it is BC breaking.
        """
        if early_stopping == "never" and length_penalty > 0.0:
            best_hypothetical_length = max_length - decoder_prompt_len
        else:
            best_hypothetical_length = cur_len - decoder_prompt_len
        best_possible_running_score = running_beam_scores[:, :1] / (
            best_hypothetical_length**length_penalty)
        worst_finished_score = torch.where(is_sent_finished, torch.min(
            beam_scores, dim=1, keepdim=True)[0], -1.0e9)
        return is_early_stop_heuristic_unsatisfied & torch.any(
            best_possible_running_score > worst_finished_score, dim=-1, keepdim=True
        )

    @staticmethod
    def _beam_search_has_unfinished_sequences(
        is_early_stop_heuristic_unsatisfied: torch.Tensor,
        is_sent_finished: torch.Tensor,
        next_token_hits_stopping_criteria: torch.Tensor,
        early_stopping: Union[bool, str],
    ):
        """
        Beam Search stopping condition -- halts the generation loop if any of these conditions becomes False
        """
        # a. Can the open beams improve the top completed scores?
        improvement_possible = torch.any(is_early_stop_heuristic_unsatisfied)

        # b. Is there still a beam without fully completed sequences? This is only relevant if early_stopping is
        # enabled, where we want to finish as soon as all beams have a completed sequence.
        exists_open_beam = ~(torch.all(is_sent_finished)
                             & (early_stopping is True))

        # c. Have we hit a stopping criteria with all running sequences and have no way to continue? e.g. we have
        # reached `max_length``
        valid_continuations = ~torch.all(next_token_hits_stopping_criteria)

        return improvement_possible & exists_open_beam & valid_continuations

    def _get_top_k_continuations(
        self,
        accumulated_log_probs: torch.Tensor,
        running_sequences: torch.Tensor,
        running_beam_indices: torch.Tensor,
        cur_len: int,
        decoder_prompt_len: int,
        do_sample: bool,
        beams_to_keep: int,
        num_beams: int,
        vocab_size: int,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取top-k候选序列"""
        if do_sample:
            probs = torch.nn.functional.softmax(accumulated_log_probs, dim=-1)
            topk_indices = torch.multinomial(probs, num_samples=beams_to_keep)
            topk_log_probs = torch.gather(
                accumulated_log_probs, 1, topk_indices)
        else:
            topk_log_probs, topk_indices = torch.topk(
                accumulated_log_probs, k=beams_to_keep, dim=-1)

        # 恢复beam索引和token索引
        topk_current_beam_indices = topk_indices // vocab_size
        topk_running_beam_indices = self._gather_beams(
            running_beam_indices, topk_current_beam_indices)
        topk_running_sequences = self._gather_beams(
            running_sequences, topk_current_beam_indices)
        topk_ids = topk_indices % vocab_size

        # 更新序列
        topk_running_sequences[:, :, cur_len] = topk_ids

        # 更新beam indices
        batch_offset = torch.arange(
            batch_size, device=topk_ids.device).view(-1, 1) * num_beams
        batch_modified_indices = topk_current_beam_indices + batch_offset
        topk_running_beam_indices[:, :, cur_len -
                                  decoder_prompt_len] = batch_modified_indices

        return topk_log_probs, topk_running_sequences, topk_running_beam_indices

    def _get_running_beams_for_next_iteration(
        self,
        topk_log_probs: torch.Tensor,
        topk_running_sequences: torch.Tensor,
        topk_running_beam_indices: torch.Tensor,
        next_token_hits_stopping_criteria: torch.Tensor,
        num_beams: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Given the top-K continuations, their scores, and whether they hit a stopping criteria, select the
        best non-finished beams to continue beam search in the next iteration.
        """
        # To prevent these just finished sequences from being used in subsequent iterations, set their log probs
        # to a very large negative value
        topk_running_log_probs = topk_log_probs + \
            next_token_hits_stopping_criteria.to(torch.float32) * -1.0e9

        next_topk_indices = torch.topk(topk_running_log_probs, k=num_beams)[1]
        running_sequences = self._gather_beams(
            topk_running_sequences, next_topk_indices)
        running_beam_scores = self._gather_beams(
            topk_running_log_probs, next_topk_indices)
        running_beam_indices = self._gather_beams(
            topk_running_beam_indices, next_topk_indices)
        return running_sequences, running_beam_scores, running_beam_indices

    def _update_finished_beams(
        self,
        sequences: torch.Tensor,
        topk_running_sequences: torch.Tensor,
        beam_scores: torch.Tensor,
        topk_log_probs: torch.Tensor,
        beam_indices: torch.Tensor,
        topk_running_beam_indices: torch.Tensor,
        is_early_stop_heuristic_unsatisfied: torch.Tensor,
        is_sent_finished: torch.Tensor,
        next_token_hits_stopping_criteria: torch.Tensor,
        top_num_beam_mask: torch.Tensor,
        num_beams: int,
        cur_len: int,
        decoder_prompt_len: int,
        length_penalty: float,
        early_stopping: Union[bool, str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the finished beams if (and only if) there are new completed sequences that have a higher score than
        the current finished sequences.
        """
        # Only the top `num_beam` sequences can be considered for the final returned sequences. Remember: the
        # remaining sequences only exist as a backup to ensure that we have at least `num_beams` sequences to
        # continue.
        did_top_num_beams_just_finished = next_token_hits_stopping_criteria & top_num_beam_mask[
            None, :]

        # Further process topk logits for the finished beams
        # - add length penalty
        topk_log_probs = topk_log_probs / \
            ((cur_len + 1 - decoder_prompt_len) ** length_penalty)
        # - make sure no scores can be added anymore if beam is full and early stopping is on
        beams_in_batch_are_full = torch.all(
            is_sent_finished, axis=-1, keepdims=True) & (early_stopping is True)
        topk_log_probs += beams_in_batch_are_full.to(torch.float32) * -1.0e9
        # - make sure no scores can be added anymore if improvement is not possible
        topk_log_probs += (~is_early_stop_heuristic_unsatisfied).to(torch.float32) * -1.0e9

        # - make sure still running sequences cannot be chosen as finalized beam
        topk_log_probs += (~did_top_num_beams_just_finished) * -1.0e9

        # Get finalized  `num_beam` sequences for the next generation step -- combine the previous finalized
        # data with the new finalized sequences (if any, non-finalized sequences have a very large negative score
        # in this step), and keep the best `num_beams` sequences.
        merged_sequences = torch.cat(
            (sequences, topk_running_sequences), dim=1)
        merged_scores = torch.cat((beam_scores, topk_log_probs), dim=1)
        merged_beam_indices = torch.cat(
            (beam_indices, topk_running_beam_indices), dim=1)
        merged_is_sent_finished = torch.cat(
            (is_sent_finished, did_top_num_beams_just_finished), dim=1)
        topk_merged_indices = torch.topk(merged_scores, k=num_beams)[1]
        sequences = self._gather_beams(merged_sequences, topk_merged_indices)
        beam_scores = self._gather_beams(merged_scores, topk_merged_indices)
        beam_indices = self._gather_beams(
            merged_beam_indices, topk_merged_indices)
        is_sent_finished = self._gather_beams(
            merged_is_sent_finished, topk_merged_indices)
        return sequences, beam_scores, beam_indices, is_sent_finished

    def beam_search_expr(
        self,
        prompts: List[Union[str, List[int]]],
        params: BeamSearchParams,
    ) -> List[BeamSearchOutput]:
        """ 移植 transformers 的 beam search 逻辑 """
        # step0：需要放在最前面完成的准备工作

        tokenizer = self.get_tokenizer()

        # vllm 自带
        beam_width = params.beam_width
        max_tokens = params.max_tokens
        temperature = params.temperature
        ignore_eos = params.ignore_eos

        # transformers 引入
        pad_token_id = tokenizer.pad_token_id
        eos_token_id = tokenizer.eos_token_id
        do_sample = params.do_sample
        early_stopping = params.early_stopping
        length_penalty = params.length_penalty
        max_length = params.max_tokens
        num_beams = params.beam_width
        num_return_sequences = params.num_return_sequences
        batch_size = len(prompts)

        # step1: encode 转换成 transformers 的代码的矩阵输入形式（str -> torch.LongTensor）

        # encode
        if isinstance(prompts[0], str):  # 字符串输入，使用tokenizer编码
            encoded_inputs = tokenizer(
                prompts,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
                truncation=True
            )
            input_ids = encoded_inputs["input_ids"]
        else:  # 已经是token ids, 确保所有序列长度一致，进行padding
            max_len = max(len(seq) for seq in prompts)
            input_ids = torch.full(
                (batch_size, max_len), pad_token_id, dtype=torch.long)
            for i, seq in enumerate(prompts):
                input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long)

        device = input_ids.device
        _, cur_len = input_ids.shape

        # step2: 移植 transformers 的 beam_search 逻辑

        # step2.1: 初始化beam search需要的变量
        vocab_size = self.llm_engine.model_config.get_vocab_size()
        input_ids = input_ids.repeat_interleave(
            num_beams, dim=0)  # 扩展输入到num_beams倍

        # 初始化running sequences
        running_sequences = torch.full(
            (batch_size, num_beams, max_length),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=device,
        )
        running_sequences[:, :, :cur_len] = self._unflatten_beam_dim(
            input_ids, batch_size, num_beams)
        sequences = running_sequences.detach().clone()

        # 初始化beam scores
        running_beam_scores = torch.zeros(
            (batch_size, num_beams), dtype=torch.float, device=device)
        running_beam_scores[:, 1:] = -1e9
        beam_scores = torch.full(
            (batch_size, num_beams), fill_value=-1e9, dtype=torch.float, device=device)

        # 初始化完成状态
        is_sent_finished = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=device)
        is_early_stop_heuristic_unsatisfied = torch.ones(
            (batch_size, 1), dtype=torch.bool, device=device)
        next_token_hits_stopping_criteria = torch.zeros(
            (batch_size, num_beams), dtype=torch.bool, device=device)

        # 初始化beam indices
        running_beam_indices = torch.full(
            (batch_size, num_beams, max_length - cur_len),
            fill_value=-1,
            dtype=torch.int32,
            device=device
        )
        beam_indices = running_beam_indices.detach().clone()

        # beam search参数
        n_eos_tokens = len(eos_token_id) if isinstance(
            eos_token_id, list) else 1
        beams_to_keep = max(2, 1 + n_eos_tokens) * num_beams
        top_num_beam_mask = torch.cat(
            (torch.ones((num_beams), dtype=torch.bool),
             torch.zeros((beams_to_keep - num_beams), dtype=torch.bool)),
            dim=0
        ).to(device)

        decoder_prompt_len = cur_len

        # step2.2: vllm generate - 移植beam search主循环
        while cur_len < max_length:
            # 准备模型输入
            flat_running_sequences = self._flatten_beam_dim(
                running_sequences[:, :, :cur_len])

            # 这里需要调用vLLM的forward方法获取logits
            # 假设self.forward接受input_ids并返回logits
            # input_ids 转换为 vllm 的 generate 输入 (torch.LongTensor -> PromptType)
            model_inputs = self._get_token_prompts(flat_running_sequences)
            model_outputs = self.generate(prompts=model_inputs, sampling_params=SamplingParams(logprobs=num_beams,
                                                                                               max_tokens=1,
                                                                                               temperature=temperature),
                                          use_tqdm=False)

            # 停止条件检查
            next_token_hits_stopping_criteria = torch.zeros(
                (batch_size, num_beams, beams_to_keep // num_beams), dtype=torch.bool, device=device
            )

            log_probs = torch.full(
                # 初始化为一个极小值
                (batch_size * num_beams, vocab_size), -1e9, dtype=torch.bfloat16)
            for batch_idx in range(batch_size):
                for beam_idx in range(num_beams):
                    idx = batch_idx * num_beams + beam_idx
                    # for cand_idx, (token_id, log_prob_obj) in enumerate(model_outputs[idx].outputs[0].logprobs[0].items()):
                    kv = list(
                        model_outputs[idx].outputs[0].logprobs[0].items())
                    for cand_idx in range(num_beams):
                        (token_id, log_prob_obj) = kv[cand_idx]
                        log_probs[idx][token_id] = log_prob_obj.logprob
            log_probs = self._unflatten_beam_dim(
                log_probs, batch_size, num_beams)
            log_probs = log_probs + running_beam_scores[:, :, None]
            log_probs = torch.reshape(
                log_probs, (batch_size, num_beams * vocab_size))

            # transformers的停止逻辑复杂得多
            if not ignore_eos:
                next_token_hits_stopping_criteria = topk_running_sequences[:,
                                                                           :, cur_len] == eos_token_id

            # 获取top-k候选
            topk_log_probs, topk_running_sequences, topk_running_beam_indices = self._get_top_k_continuations(
                accumulated_log_probs=log_probs,
                running_sequences=running_sequences,
                running_beam_indices=running_beam_indices,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                do_sample=do_sample,
                beams_to_keep=beams_to_keep,
                num_beams=num_beams,
                vocab_size=vocab_size,
                batch_size=batch_size,
            )

            # 获取下一个iteration的运行beam
            running_sequences, running_beam_scores, running_beam_indices = self._get_running_beams_for_next_iteration(
                topk_log_probs=topk_log_probs,
                topk_running_sequences=topk_running_sequences,
                topk_running_beam_indices=topk_running_beam_indices,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                num_beams=num_beams,
            )

            # 更新完成的beam
            sequences, beam_scores, beam_indices, is_sent_finished = self._update_finished_beams(
                sequences=sequences,
                topk_running_sequences=topk_running_sequences,
                beam_scores=beam_scores,
                topk_log_probs=topk_log_probs,
                beam_indices=beam_indices,
                topk_running_beam_indices=topk_running_beam_indices,
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                is_sent_finished=is_sent_finished,
                next_token_hits_stopping_criteria=next_token_hits_stopping_criteria,
                top_num_beam_mask=top_num_beam_mask,
                num_beams=num_beams,
                cur_len=cur_len,
                decoder_prompt_len=decoder_prompt_len,
                length_penalty=length_penalty,
                early_stopping=early_stopping,
            )

            cur_len += 1

            # 检查是否所有序列都已完成
            is_early_stop_heuristic_unsatisfied = self._check_early_stop_heuristic(
                is_early_stop_heuristic_unsatisfied=is_early_stop_heuristic_unsatisfied,
                running_beam_scores=running_beam_scores,
                beam_scores=beam_scores,
                is_sent_finished=is_sent_finished,
                cur_len=cur_len,
                max_length=max_length,
                decoder_prompt_len=decoder_prompt_len,
                early_stopping=early_stopping,
                length_penalty=length_penalty,
            )

            if not self._beam_search_has_unfinished_sequences(is_early_stop_heuristic_unsatisfied,
                                                              is_sent_finished,
                                                              next_token_hits_stopping_criteria,
                                                              early_stopping):
                break

        # step3：decode 输出(torch.Tensor -> List[RequestOutput])
        sequences = self._flatten_beam_dim(
            sequences[:, :num_return_sequences, :])
        beam_scores = self._flatten_beam_dim(
            beam_scores[:, :num_return_sequences])

        # 转换输出格式
        outputs = []
        for i in range(batch_size):
            batch_outputs = []
            for j in range(num_return_sequences):
                idx = i * num_return_sequences + j
                seq = sequences[idx].tolist()
                if pad_token_id in seq:  # 移除padding tokens
                    seq = seq[:seq.index(pad_token_id)]
                if eos_token_id in seq:  # 移除eos token之后的部分
                    eos_pos = seq.index(eos_token_id)
                    seq = seq[:eos_pos + 1]
                seq.append(eos_token_id)  # 末尾加上单个终止符

                score = beam_scores[idx].item() if idx < len(
                    beam_scores) else 0.0

                batch_outputs.append(BeamSearchSequence(
                    text=tokenizer.decode(seq),
                    tokens=seq,
                    cum_logprob=score
                ))
            outputs.append(BeamSearchOutput(sequences=batch_outputs))
        return outputs

    def chat(
        self,
        messages: Union[List[ChatCompletionMessageParam],
                        List[List[ChatCompletionMessageParam]]],
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        chat_template: Optional[str] = None,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> List[RequestOutput]:
        """
        Generate responses for a chat conversation.

        The chat conversation is converted into a text prompt using the
        tokenizer and calls the :meth:`generate` method to generate the
        responses.

        Multi-modal inputs can be passed in the same way you would pass them
        to the OpenAI API.

        Args:
            messages: A list of conversations or a single conversation. 
                - Each conversation is represented as a list of messages.
                - Each message is a dictionary with 'role' and 'content' keys.
            sampling_params: The sampling parameters for text generation.
                If None, we use the default sampling parameters. When it
                is a single value, it is applied to every prompt. When it
                is a list, the list must have the same length as the
                prompts and it is paired one by one with the prompt.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            chat_template: The template to use for structuring the chat.
              If not provided, the model's default chat template will be used.
            add_generation_prompt: If True, adds a generation template
                to each message.
            continue_final_message: If True, continues the final message in
                the conversation instead of starting a new one. Cannot be `True`
                if `add_generation_prompt` is also `True`.

        Returns:
            A list of ``RequestOutput`` objects containing the generated
            responses in the same order as the input messages.
        """
        list_of_messages: List[List[ChatCompletionMessageParam]]

        # Handle multi and single conversations
        if is_list_of(messages, list):
            # messages is List[List[...]]
            list_of_messages = messages
        else:
            # messages is List[...]
            list_of_messages = [messages]

        prompts: List[Union[TokensPrompt, TextPrompt]] = []

        for msgs in list_of_messages:
            tokenizer = self.get_tokenizer()
            model_config = self.llm_engine.get_model_config()

            conversation, mm_data = parse_chat_messages(
                msgs, model_config, tokenizer)

            prompt_data: Union[str, List[int]]
            if isinstance(tokenizer, MistralTokenizer):
                prompt_data = apply_mistral_chat_template(
                    tokenizer,
                    messages=msgs,
                    chat_template=chat_template,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                )
            else:
                prompt_data = apply_hf_chat_template(
                    tokenizer,
                    conversation=conversation,
                    chat_template=chat_template,
                    add_generation_prompt=add_generation_prompt,
                    continue_final_message=continue_final_message,
                    tools=tools,
                )

            prompt: Union[TokensPrompt, TextPrompt]
            if is_list_of(prompt_data, int):
                prompt = TokensPrompt(prompt_token_ids=prompt_data)
            else:
                prompt = TextPrompt(prompt=prompt_data)

            if mm_data is not None:
                prompt["multi_modal_data"] = mm_data

            prompts.append(prompt)

        return self.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

    @overload  # LEGACY: single (prompt + optional token ids)
    def encode(
        self,
        prompts: str,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[List[int]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: multi (prompt + optional token ids)
    def encode(
        self,
        prompts: List[str],
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: single (token ids + optional prompt)
    def encode(
        self,
        prompts: Optional[str] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: List[int],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: multi (token ids + optional prompt)
    def encode(
        self,
        prompts: Optional[List[str]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        *,
        prompt_token_ids: List[List[int]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload  # LEGACY: single or multi token ids [pos-only]
    def encode(
        self,
        prompts: None,
        pooling_params: None,
        prompt_token_ids: Union[List[int], List[List[int]]],
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @overload
    def encode(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        /,
        *,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
    ) -> List[EmbeddingRequestOutput]:
        ...

    @deprecate_kwargs(
        "prompt_token_ids",
        is_deprecated=lambda: LLM.DEPRECATE_LEGACY,
        additional_message="Please use the 'prompts' parameter instead.",
    )
    def encode(
        self,
        prompts: Union[Union[PromptType, Sequence[PromptType]],
                       Optional[Union[str, List[str]]]] = None,
        pooling_params: Optional[Union[PoolingParams,
                                       Sequence[PoolingParams]]] = None,
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[Union[List[LoRARequest], LoRARequest]] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
    ) -> List[EmbeddingRequestOutput]:
        """Generates the completions for the input prompts.

        This class automatically batches the given prompts, considering
        the memory constraint. For the best performance, put all of your prompts
        into a single list and pass it to this method.

        Args:
            prompts: The prompts to the LLM. You may pass a sequence of prompts
                for batch inference. See :class:`~vllm.inputs.PromptType`
                for more details about the format of each prompts.
            pooling_params: The pooling parameters for pooling. If None, we
                use the default pooling parameters.
            use_tqdm: Whether to use tqdm to display the progress bar.
            lora_request: LoRA request to use for generation, if any.
            prompt_adapter_request: Prompt Adapter request to use for
                generation, if any.

        Returns:
            A list of `EmbeddingRequestOutput` objects containing the
            generated embeddings in the same order as the input prompts.

        Note:
            Using ``prompts`` and ``prompt_token_ids`` as keyword parameters is
            considered legacy and may be deprecated in the future. You should
            instead pass them via the ``inputs`` parameter.
        """
        if not self.llm_engine.model_config.embedding_mode:
            raise ValueError(
                "LLM.encode() is only supported for embedding models (XModel)."
            )

        if prompt_token_ids is not None:
            parsed_prompts = self._convert_v1_inputs(
                prompts=cast(Optional[Union[str, List[str]]], prompts),
                prompt_token_ids=prompt_token_ids,
            )
        else:
            parsed_prompts = cast(Union[PromptType, Sequence[PromptType]],
                                  prompts)

        if pooling_params is None:
            # Use default pooling params.
            pooling_params = PoolingParams()

        self._validate_and_add_requests(
            prompts=parsed_prompts,
            params=pooling_params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
        )

        outputs = self._run_engine(use_tqdm=use_tqdm)
        return LLMEngine.validate_outputs(outputs, EmbeddingRequestOutput)

    def start_profile(self) -> None:
        self.llm_engine.start_profile()

    def stop_profile(self) -> None:
        self.llm_engine.stop_profile()

    # LEGACY
    def _convert_v1_inputs(
        self,
        prompts: Optional[Union[str, List[str]]],
        prompt_token_ids: Optional[Union[List[int], List[List[int]]]],
    ):
        # skip_tokenizer_init is now checked in engine

        if prompts is not None:
            prompts = [p["content"] for p in parse_and_batch_prompt(prompts)]
        if prompt_token_ids is not None:
            prompt_token_ids = [
                p["content"] for p in parse_and_batch_prompt(prompt_token_ids)
            ]

        num_requests = None
        if prompts is not None:
            num_requests = len(prompts)
        if prompt_token_ids is not None:
            if (num_requests is not None
                    and num_requests != len(prompt_token_ids)):
                raise ValueError("The lengths of prompts and prompt_token_ids "
                                 "must be the same.")

            num_requests = len(prompt_token_ids)
        if num_requests is None:
            raise ValueError("Either prompts or prompt_token_ids must be "
                             "provided.")

        parsed_prompts: List[PromptType] = []
        for i in range(num_requests):
            item: PromptType

            if prompts is not None:
                item = TextPrompt(prompt=prompts[i])
            elif prompt_token_ids is not None:
                item = TokensPrompt(prompt_token_ids=prompt_token_ids[i])
            else:
                raise AssertionError

            parsed_prompts.append(item)

        return parsed_prompts

    def _validate_and_add_requests(
        self,
        prompts: Union[PromptType, Sequence[PromptType]],
        params: Union[SamplingParams, Sequence[SamplingParams], PoolingParams,
                      Sequence[PoolingParams]],
        lora_request: Optional[Union[Sequence[LoRARequest], LoRARequest]],
        prompt_adapter_request: Optional[PromptAdapterRequest],
        guided_options: Optional[GuidedDecodingRequest] = None,
        priority: Optional[List[int]] = None,
    ) -> None:
        if guided_options is not None:
            warnings.warn(
                "guided_options_request is deprecated, use "
                "SamplingParams.guided_decoding instead",
                DeprecationWarning,
                stacklevel=2,
            )

        if isinstance(prompts, (str, dict)):
            # Convert a single prompt to a list.
            prompts = [prompts]

        num_requests = len(prompts)
        if isinstance(params, list) and len(params) != num_requests:
            raise ValueError("The lengths of prompts and params "
                             "must be the same.")
        if isinstance(lora_request,
                      list) and len(lora_request) != num_requests:
            raise ValueError("The lengths of prompts and lora_request "
                             "must be the same.")

        for sp in params if isinstance(params, list) else (params, ):
            if isinstance(sp, SamplingParams):
                self._add_guided_params(sp, guided_options)

                # We only care about the final output
                sp.output_kind = RequestOutputKind.FINAL_ONLY

        # Add requests to the engine.
        for i, prompt in enumerate(prompts):
            self._add_request(
                prompt,
                params[i] if isinstance(params, Sequence) else params,
                lora_request=lora_request[i] if isinstance(
                    lora_request, Sequence) else lora_request,
                prompt_adapter_request=prompt_adapter_request,
                priority=priority[i] if priority else 0,
            )

    def _add_request(
        self,
        prompt: PromptType,
        params: Union[SamplingParams, PoolingParams],
        lora_request: Optional[LoRARequest] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None,
        priority: int = 0,
    ) -> None:
        request_id = str(next(self.request_counter))
        self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
            prompt_adapter_request=prompt_adapter_request,
            priority=priority,
        )

    def _add_guided_params(
            self,
            params: SamplingParams,
            guided_options: Optional[GuidedDecodingRequest] = None):
        if guided_options is None:
            return params

        if params.guided_decoding is not None:
            raise ValueError("Cannot set both guided_options_request and"
                             "params.guided_decoding.")

        params.guided_decoding = GuidedDecodingParams(
            json=guided_options.guided_json,
            regex=guided_options.guided_regex,
            choice=guided_options.guided_choice,
            grammar=guided_options.guided_grammar,
            json_object=guided_options.guided_json_object,
            backend=guided_options.guided_decoding_backend,
            whitespace_pattern=guided_options.guided_whitespace_pattern)
        return params

    def _run_engine(
            self, *, use_tqdm: bool
    ) -> List[Union[RequestOutput, EmbeddingRequestOutput]]:
        # Initialize tqdm.
        if use_tqdm:
            num_requests = self.llm_engine.get_num_unfinished_requests()
            pbar = tqdm(
                total=num_requests,
                desc="Processed prompts",
                dynamic_ncols=True,
                postfix=(f"est. speed input: {0:.2f} toks/s, "
                         f"output: {0:.2f} toks/s"),
            )

        # Run the engine.
        outputs: List[Union[RequestOutput, EmbeddingRequestOutput]] = []
        total_in_toks = 0
        total_out_toks = 0
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                if output.finished:
                    outputs.append(output)
                    if use_tqdm:
                        if isinstance(output, RequestOutput):
                            # Calculate tokens only for RequestOutput
                            assert output.prompt_token_ids is not None
                            total_in_toks += len(output.prompt_token_ids)
                            in_spd = total_in_toks / \
                                pbar.format_dict["elapsed"]
                            total_out_toks += sum(
                                len(stp.token_ids) for stp in output.outputs)
                            out_spd = (total_out_toks /
                                       pbar.format_dict["elapsed"])
                            pbar.postfix = (
                                f"est. speed input: {in_spd:.2f} toks/s, "
                                f"output: {out_spd:.2f} toks/s")
                        pbar.update(1)

        if use_tqdm:
            pbar.close()
        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))

    def _is_encoder_decoder_model(self):
        return self.llm_engine.is_encoder_decoder_model()

    def _is_embedding_model(self):
        return self.llm_engine.is_embedding_model()
