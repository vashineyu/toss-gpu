import GPUtil


def get_gpus_on_demand(
    min_request: int = 1,
    max_request: int = 1,
    memory_lb: int = 1024
):
    available_gpus = GPUtil.getAvailability(
        GPUtil.getGPUs(),
        memoryFree=memory_lb,
    )
    if len(available_gpus) < min_request:
        return []

    available_gpu_ids = [
        i for i, is_available in enumerate(available_gpus) if is_available
    ][:max_request]
    return available_gpu_ids

