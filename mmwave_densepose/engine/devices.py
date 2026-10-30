"""Visible CUDA devices and explicit task-device selection."""
from __future__ import annotations

import re


def discover_devices():
    """Report CUDA indices visible to this process without changing visibility."""
    import torch
    try:
        if not torch.cuda.is_available():
            return {"cuda_available": False, "devices": []}
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append({"id": index, "name": str(properties.name),
                            "total_memory_bytes": int(properties.total_memory)})
        return {"cuda_available": bool(devices), "devices": devices}
    except (OSError, RuntimeError, AssertionError) as error:
        return {"cuda_available": False, "devices": [],
                "error": f"CUDA device discovery failed: {error}"}


def canonical_gpu_ids(device, gpu_ids, *, action=None):
    """Validate configuration syntax without querying machine-specific hardware."""
    if gpu_ids is None:
        return []
    if not isinstance(gpu_ids, (list, tuple)):
        raise ValueError("gpu_ids must be a list of CUDA device indices")
    result = list(gpu_ids)
    if any(isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in result):
        raise ValueError("GPU indices must be nonnegative integers")
    if len(set(result)) != len(result):
        raise ValueError("Select each GPU only once")
    if result:
        if action is not None and action != "train" and len(result) > 1:
            raise ValueError("Evaluation and inference use one GPU; select a single GPU")
        if str(device) not in {"auto", "cuda"} and not re.fullmatch(r"cuda:\d+", str(device)):
            raise ValueError("GPU selection requires a CUDA device")
        if str(device).startswith("cuda:") and int(str(device).split(":", 1)[1]) != result[0]:
            raise ValueError("The CUDA device must match the first selected GPU")
    return result


def validate_gpu_selection(action, device, gpu_ids, batch_size=None):
    """Check configured CUDA indices against the local process-visible inventory."""
    result = canonical_gpu_ids(device, gpu_ids, action=action)
    cuda_device = str(device) == "cuda" or bool(re.fullmatch(r"cuda:\d+", str(device)))
    if not result and not cuda_device:
        return result
    inventory = discover_devices()
    if not inventory["cuda_available"]:
        raise ValueError(inventory.get("error") or "CUDA is unavailable; select CPU")
    available = {item["id"] for item in inventory["devices"]}
    requested = result or [int(str(device).split(":", 1)[1]) if ":" in str(device) else 0]
    if any(index not in available for index in requested):
        raise ValueError("A selected GPU is not available to this process; refresh the GPU list")
    if len(result) > 1 and batch_size is not None:
        if isinstance(batch_size, bool) or int(batch_size) < 2 * len(result):
            raise ValueError("Multi-GPU training requires a batch size of at least two samples per selected GPU")
    return result


__all__ = ["discover_devices", "canonical_gpu_ids", "validate_gpu_selection"]
