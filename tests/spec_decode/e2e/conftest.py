import asyncio
import os
from itertools import cycle
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pytest
import ray
import torch

from aphrodite import LLM
from aphrodite.common.outputs import RequestOutput
from aphrodite.common.sampling_params import SamplingParams
from aphrodite.common.sequence import Logprob
from aphrodite.common.utils import Counter, random_uuid
from aphrodite.engine.args_tools import AsyncEngineArgs
from aphrodite.engine.async_aphrodite import AsyncAphrodite
from aphrodite.lora.request import LoRARequest
from aphrodite.modeling.utils import set_random_seed
from aphrodite.multimodal import MultiModalDataDict
from aphrodite.prompt_adapter.request import PromptAdapterRequest

from ...conftest import cleanup
from ...utils import wait_for_gpu_memory_to_clear


class AsyncLLM:
    """AsyncLLM

    Note: Current LLM class in aphrodite don't support async mode, for test
    purpose, we implement async one in here. Maybe we could move to
    aphrodite/endpoints/llm.py in future.

    Below AsyncLLM is directly borrow from aphrodite/endpoints/llm.py with
    changes to make to work in async mode.
    """

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
        swap_space: int = 4,
        enforce_eager: bool = False,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        **kwargs,
    ) -> None:
        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True

        # Needed to engine_use_ray works as a deprecated feature,
        # otherwise the following constructor will raise an exception
        os.environ["APHRODITE_ALLOW_ENGINE_USE_RAY"] = "1"

        engine_args = AsyncEngineArgs(
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
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            # For now use ray for the distributed back-end, since
            # we rely on the use of engine_use_ray=True to avoid
            # reinitializing CUDA in the same process (driver worker)
            engine_use_ray=True,
            distributed_executor_backend="ray",
            disable_custom_all_reduce=disable_custom_all_reduce,
            **kwargs,
        )
        self.request_counter = Counter()
        self.llm_engine = AsyncAphrodite.from_engine_args(engine_args)

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Union[SamplingParams,
                                        List[SamplingParams]]] = None,
        prompt_token_ids: Optional[List[List[int]]] = None,
        use_tqdm: bool = True,
        lora_request: Optional[LoRARequest] = None,
        multi_modal_data: Optional[MultiModalDataDict] = None,
        prompt_adapter_request: Optional[PromptAdapterRequest] = None
    ) -> List[RequestOutput]:

        if prompts is None:
            raise ValueError("prompts must be provided.")
        if isinstance(prompts, str):
            # Convert a single prompt to a list.
            prompts = [prompts]

        if prompts is not None:
            num_requests = len(prompts)

        if sampling_params is None:
            # Use default sampling params.
            sampling_params = SamplingParams()

        elif isinstance(sampling_params,
                        list) and len(sampling_params) != num_requests:
            raise ValueError("The lengths of prompts and "
                             "sampling_params must be the same.")

        async def get_output(prompt, sampling_param) -> RequestOutput:
            request_id = random_uuid()
            results_generator = self.llm_engine.generate(
                prompt, sampling_param, request_id)
            final_output = None
            async for request_output in results_generator:
                final_output = request_output
            assert final_output is not None
            return final_output

        outputs: List[RequestOutput] = []
        try:
            for i in range(num_requests):
                prompt = prompts[i] if prompts is not None else None
                params = sampling_params[i] if isinstance(
                    sampling_params, Sequence) else sampling_params
                res = asyncio.run(get_output(prompt, params))
                outputs.append(res)
        finally:
            ray.shutdown()
        return outputs


@pytest.fixture
def baseline_llm_generator(request, common_llm_kwargs,
                           per_test_common_llm_kwargs, baseline_llm_kwargs,
                           seed):
    return create_llm_generator("baseline", request, common_llm_kwargs,
                                per_test_common_llm_kwargs,
                                baseline_llm_kwargs, seed)


@pytest.fixture
def test_llm_generator(request, common_llm_kwargs, per_test_common_llm_kwargs,
                       test_llm_kwargs, seed):
    return create_llm_generator("test", request, common_llm_kwargs,
                                per_test_common_llm_kwargs, test_llm_kwargs,
                                seed)


def create_llm_generator(baseline_or_test, request, common_llm_kwargs,
                         per_test_common_llm_kwargs, distinct_llm_kwargs,
                         seed):
    kwargs = {
        **common_llm_kwargs,
        **per_test_common_llm_kwargs,
        **distinct_llm_kwargs,
    }
    test_name = request.node.name

    model = kwargs["model"]
    draft_model = kwargs.get("speculative_model", None)
    same_draft_target_model = (draft_model is not None
                               and draft_model == model)

    def generator_inner():

        wait_for_gpu_memory_to_clear(
            devices=list(range(torch.cuda.device_count())),
            threshold_bytes=2 * 2**30,
            timeout_s=60,
        )

        use_async = False
        if "use_async" in kwargs:
            use_async = kwargs.pop("use_async")
        print(f'{use_async=}')

        print(f'Creating {baseline_or_test=} LLM for {test_name=}. {kwargs=}')
        llm = AsyncLLM(**kwargs) if use_async else LLM(**kwargs)

        # Override logging interval to 0 for spec decode test run to
        # log all metrics in time.
        if (baseline_or_test == "test" and not use_async
                and llm.llm_engine.log_stats):
            for sate_logger in llm.llm_engine.stat_loggers.values():
                sate_logger.local_interval = 0
        if seed is not None:
            set_random_seed(seed)

        yield llm
        del llm
        cleanup()

    def generator_outer():
        for llm in generator_inner():
            yield llm
            del llm

    # Set an attribute to the generator_outer function to allow us to
    # determine whether to further check the acceptance rate in tests.
    generator_outer.same_draft_target_model = same_draft_target_model  # type: ignore
    return generator_outer


def maybe_assert_ngram_worker(llm):
    # Verify the proposer worker is ngram if ngram is specified.
    if (not isinstance(llm, AsyncLLM)
            and llm.llm_engine.speculative_config is not None
            and llm.llm_engine.speculative_config.ngram_prompt_lookup_max > 0):
        from aphrodite.spec_decode.ngram_worker import NGramWorker
        assert isinstance(
            llm.llm_engine.model_executor.driver_worker.proposer_worker,
            NGramWorker)


def get_output_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> Tuple[List[str], List[List[int]], float]:
    tokens: List[str] = []
    token_ids: List[List[int]] = []
    acceptance_rate: float = -1.0
    for llm in llm_generator():
        maybe_assert_ngram_worker(llm)

        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)

        token_ids = [output.outputs[0].token_ids for output in outputs]
        tokens = [output.outputs[0].text for output in outputs]

        # Fetch acceptance rate if logging is enabled.
        if stat_loggers := getattr(llm.llm_engine, "stat_loggers", None):
            stat_logger = stat_loggers["prometheus"]
            acceptance_rate = (stat_logger.metrics.
                               gauge_spec_decode_draft_acceptance_rate.labels(
                                   **stat_logger.labels)._value.get())
        del llm

    return tokens, token_ids, acceptance_rate


def get_logprobs_from_llm_generator(
        llm_generator, prompts,
        sampling_params) -> List[List[Dict[int, Logprob]]]:
    """Returns a dict of (token_id: Logprob) for each generated position, for
    each sequence in the batch.
    """
    for llm in llm_generator():
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        logprobs = [output.outputs[0].logprobs[:] for output in outputs]
        del llm

    return logprobs


def run_greedy_equality_correctness_test(baseline_llm_generator,
                                         test_llm_generator,
                                         batch_size,
                                         max_output_len,
                                         force_output_len: bool,
                                         print_tokens: bool = False,
                                         ensure_all_accepted: bool = False):
    """Helper method that compares the outputs of both the baseline LLM and
    the test LLM. It asserts greedy equality, e.g. that the outputs are exactly
    the same when temperature is zero.
    """

    run_equality_correctness_test(baseline_llm_generator,
                                  test_llm_generator,
                                  batch_size,
                                  max_output_len,
                                  force_output_len,
                                  temperature=0.0,
                                  seeded=False,
                                  print_tokens=print_tokens,
                                  ensure_all_accepted=ensure_all_accepted)


def run_equality_correctness_test(
        baseline_llm_generator,
        test_llm_generator,
        batch_size,
        max_output_len,
        force_output_len: bool,
        temperature: float,
        seeded: bool,
        print_tokens: bool = False,
        ensure_all_accepted: bool = False,
        expected_acceptance_rate: Optional[float] = None):
    """Helper method that compares the outputs of both the baseline LLM and
    the test LLM. It asserts greedy equality, e.g. that the outputs are exactly
    the same when temperature is zero (or when temperature is > 0 and seeded).
    """

    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
        "San Francisco is know for its",
        "Facebook was created in 2004 by",
        "Curious George is a",
        "Python 3.11 brings improvements to its",
    ]

    prompts = [prompt for prompt, _ in zip(cycle(prompts), range(batch_size))]

    # If the test requires that we generated max_output_len tokens, then set the
    # sampling params to ignore eos token.
    ignore_eos = force_output_len

    if seeded:
        sampling_params = [
            SamplingParams(
                max_tokens=max_output_len,
                ignore_eos=ignore_eos,
                temperature=temperature,
                seed=i,
            ) for i in range(len(prompts))
        ]
    else:
        sampling_params = SamplingParams(
            max_tokens=max_output_len,
            ignore_eos=ignore_eos,
            temperature=temperature,
        )

    (spec_batch_tokens, spec_batch_token_ids,
     acceptance_rate) = get_output_from_llm_generator(test_llm_generator,
                                                      prompts, sampling_params)

    (baseline_batch_tokens, baseline_batch_token_ids,
     _) = get_output_from_llm_generator(baseline_llm_generator, prompts,
                                        sampling_params)

    assert len(baseline_batch_token_ids) == len(prompts)
    assert len(spec_batch_token_ids) == len(prompts)

    for i, (baseline_token_ids, baseline_tokens, spec_token_ids,
            spec_tokens) in enumerate(
                zip(baseline_batch_token_ids, baseline_batch_tokens,
                    spec_batch_token_ids, spec_batch_tokens)):
        if print_tokens:
            print(f'{i=} {baseline_tokens=}')
            print(f'{i=}     {spec_tokens=}')
        print(f'{i=} {baseline_token_ids=}')
        print(f'{i=}     {spec_token_ids=}')
        assert baseline_token_ids == spec_token_ids

    print(f'{acceptance_rate=}')
    if ensure_all_accepted:
        assert acceptance_rate == 1.0
    if expected_acceptance_rate is not None:
        assert acceptance_rate >= expected_acceptance_rate - 1e-2
