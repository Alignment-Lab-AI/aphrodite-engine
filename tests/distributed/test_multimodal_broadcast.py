"""Compare the outputs of HF and distributed Aphrodite when using greedy
sampling.

Run:
```sh
pytest -s -v test_multimodal_broadcast.py
```
"""

import pytest

from aphrodite.common.utils import cuda_device_count_stateless

from ..utils import fork_new_process_for_each_test


@pytest.mark.skipif(cuda_device_count_stateless() < 2,
                    reason="Need at least 2 GPUs to run the test.")
@pytest.mark.parametrize("model, distributed_executor_backend", [
    ("llava-hf/llava-1.5-7b-hf", "ray"),
    ("llava-hf/llava-v1.6-mistral-7b-hf", "ray"),
    ("facebook/chameleon-7b", "ray"),
    ("llava-hf/llava-1.5-7b-hf", "mp"),
    ("llava-hf/llava-v1.6-mistral-7b-hf", "mp"),
    ("facebook/chameleon-7b", "mp"),
])
@fork_new_process_for_each_test
def test_models(hf_runner, aphrodite_runner, image_assets, model: str,
                distributed_executor_backend: str) -> None:

    dtype = "half"
    max_tokens = 5
    num_logprobs = 5
    tensor_parallel_size = 2

    if model.startswith("llava-hf/llava-1.5"):
        from ..models.test_llava import models, run_test
    elif model.startswith("llava-hf/llava-v1.6"):
        from ..models.test_llava_next import run_test  # type: ignore[no-redef]
        from ..models.test_llava_next import models
    elif model.startswith("facebook/chameleon"):
        from ..models.test_chameleon import run_test  # type: ignore[no-redef]
        from ..models.test_chameleon import models
    else:
        raise NotImplementedError(f"Unsupported model: {model}")

    run_test(
        hf_runner,
        aphrodite_runner,
        image_assets,
        model=models[0],
        # So that LLaVA-NeXT processor may return nested list
        size_factors=[0.25, 0.5, 1.0],
        dtype=dtype,
        max_tokens=max_tokens,
        num_logprobs=num_logprobs,
        tensor_parallel_size=tensor_parallel_size,
        distributed_executor_backend=distributed_executor_backend,
    )
