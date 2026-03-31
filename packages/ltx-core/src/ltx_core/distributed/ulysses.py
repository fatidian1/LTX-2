from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist

_STATE: "UlyssesState | None" = None


@dataclass(frozen=True)
class UlyssesState:
    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    device_mesh: object
    attention_callable: object


def _require_xfuser() -> tuple[object, object, object, object]:
    try:
        from xfuser.core.distributed import (
            get_sp_group,
            init_distributed_environment,
            initialize_model_parallel,
        )
        from xfuser.core.long_ctx_attention import xFuserLongContextAttention
        from yunchang.kernels import AttnType
    except ImportError as exc:
        raise RuntimeError(
            "Ulysses multi-GPU requires the `xfuser` and `yunchang` packages to be installed."
        ) from exc
    return get_sp_group, init_distributed_environment, initialize_model_parallel, (xFuserLongContextAttention, AttnType)


def _build_attention(attn_type: str = "FA", sync_ulysses: bool = False) -> object:
    _, _, _, (xFuserLongContextAttention, AttnType) = _require_xfuser()
    attn_aliases = {
        "TORCH": "TORCH_EFFICIENT",
        "PYTORCH": "TORCH_EFFICIENT",
        "FLASH_ATTENTION": "FA",
        "FLASH_ATTENTION_2": "FA",
        "FLASH_ATTENTION_3": "FA3",
    }
    resolved_attn_type = attn_aliases.get(attn_type, attn_type)
    try:
        xfuser_attention = xFuserLongContextAttention(
            use_sync=sync_ulysses,
            attn_type=AttnType[resolved_attn_type],
        )
    except KeyError as exc:
        available = ", ".join(member.name for member in AttnType)
        raise RuntimeError(
            f"Unsupported Ulysses attention type '{attn_type}'. "
            f"Resolved value '{resolved_attn_type}' is not in AttnType. Available: {available}"
        ) from exc

    def _attention(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is not None:
            raise ValueError("Masked attention is not supported with Ulysses sequence parallelism")

        batch_size, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (
            tensor.view(batch_size, -1, heads, dim_head).transpose(1, 2)
            for tensor in (q, k, v)
        )
        out = xfuser_attention(
            None,
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        ).transpose(1, 2)
        return out.transpose(1, 2).reshape(batch_size, -1, heads * dim_head)

    return _attention


def is_ulysses_enabled() -> bool:
    return _STATE is not None


def is_primary_rank() -> bool:
    return _STATE is None or _STATE.rank == 0


def initialize_ulysses(num_gpus: int, *, attn_type: str = "FA", sync_ulysses: bool = False) -> UlyssesState | None:
    global _STATE

    if num_gpus <= 1:
        return None

    if _STATE is not None:
        if _STATE.world_size != num_gpus:
            raise RuntimeError(f"Ulysses already initialized for {_STATE.world_size} GPUs, got {num_gpus}")
        return _STATE

    if not torch.cuda.is_available():
        raise RuntimeError("Ulysses multi-GPU requires CUDA")

    missing_env = [
        name for name in ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK") if name not in os.environ
    ]
    if not dist.is_initialized() and missing_env:
        raise RuntimeError(
            "Ulysses multi-GPU requires a distributed launcher context. "
            f"Missing environment variables: {', '.join(missing_env)}. "
            "Use the CLI entrypoint with `--num-gpus N`, `torchrun`, or your own `torch.multiprocessing.spawn` setup."
        )

    world_size = int(os.environ.get("WORLD_SIZE", num_gpus))
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if world_size != num_gpus:
        raise RuntimeError(f"Expected WORLD_SIZE={num_gpus} for Ulysses, got {world_size}")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if not dist.is_initialized():
        dist.init_process_group(
            "nccl",
            rank=rank,
            world_size=world_size,
            timeout=timedelta(minutes=10),
        )

    _, init_distributed_environment, initialize_model_parallel, _ = _require_xfuser()
    init_distributed_environment(rank=rank, world_size=world_size)
    initialize_model_parallel(
        sequence_parallel_degree=num_gpus,
        classifier_free_guidance_degree=1,
        ring_degree=1,
        ulysses_degree=num_gpus,
    )
    device_mesh = dist.device_mesh.init_device_mesh("cuda", mesh_shape=(world_size,))

    _STATE = UlyssesState(
        world_size=world_size,
        rank=rank,
        local_rank=local_rank,
        device=device,
        device_mesh=device_mesh,
        attention_callable=_build_attention(attn_type=attn_type, sync_ulysses=sync_ulysses),
    )
    return _STATE


def destroy_ulysses() -> None:
    global _STATE
    if dist.is_initialized():
        dist.destroy_process_group()
    _STATE = None


def _sequence_group() -> object:
    if _STATE is None:
        raise RuntimeError("Ulysses is not initialized")
    get_sp_group, _, _, _ = _require_xfuser()
    return get_sp_group()


def _pad_tensor(tensor: torch.Tensor, dim: int) -> tuple[torch.Tensor, int]:
    if _STATE is None:
        return tensor, tensor.size(dim)

    orig_size = tensor.size(dim)
    pad = (_STATE.world_size - orig_size % _STATE.world_size) % _STATE.world_size
    if pad == 0:
        return tensor, orig_size

    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad
    padded = torch.cat([tensor, torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)], dim=dim)
    return padded, orig_size


def shard_tensor(tensor: torch.Tensor, *, dim: int = 1) -> tuple[torch.Tensor, int]:
    if _STATE is None:
        return tensor, tensor.size(dim)

    tensor, orig_size = _pad_tensor(tensor, dim=dim)
    shard = torch.chunk(tensor, _STATE.world_size, dim=dim)[_STATE.rank]
    return shard, orig_size


def gather_tensor(tensor: torch.Tensor, orig_size: int, *, dim: int = 1) -> torch.Tensor:
    if _STATE is None:
        return tensor

    gathered = _sequence_group().all_gather(tensor.contiguous(), dim=dim)
    return gathered.narrow(dim, 0, orig_size)


def shard_rotary_embeddings(
    freqs_cis: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[tuple[torch.Tensor, torch.Tensor] | None, int | None]:
    if freqs_cis is None or _STATE is None:
        return freqs_cis, None

    cos, sin = freqs_cis
    dim = 2 if cos.ndim == 4 else 1
    cos, orig_size = shard_tensor(cos, dim=dim)
    sin, _ = shard_tensor(sin, dim=dim)
    return (cos, sin), orig_size


def ulysses_attention_callable() -> object | None:
    return None if _STATE is None else _STATE.attention_callable


def ulysses_device_mesh() -> object | None:
    return None if _STATE is None else _STATE.device_mesh
