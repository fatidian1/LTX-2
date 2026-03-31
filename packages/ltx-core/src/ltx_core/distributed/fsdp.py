from __future__ import annotations

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor, distribute_tensor

from ltx_core.distributed.ulysses import ulysses_device_mesh
from ltx_core.model.transformer import LTXModel


def _distribute_full_state_dict(
    model: LTXModel,
    model_state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor | torch.nn.Parameter]:
    sharded_meta_state_dict = model.state_dict()
    param_requires_grad = {name: param.requires_grad for name, param in model.named_parameters()}
    target_device = torch.device("cuda", torch.cuda.current_device())
    distributed_state_dict: dict[str, torch.Tensor | torch.nn.Parameter] = {}

    for name, full_tensor in model_state_dict.items():
        template = sharded_meta_state_dict.get(name)
        if template is None:
            continue

        if isinstance(template, DTensor):
            value = distribute_tensor(full_tensor, template.device_mesh, template.placements)
        else:
            value = full_tensor.to(
                device=target_device if getattr(template, "device", torch.device("meta")).type == "meta" else template.device,
                dtype=template.dtype,
            )

        if name in param_requires_grad:
            distributed_state_dict[name] = torch.nn.Parameter(value, requires_grad=param_requires_grad[name])
        else:
            distributed_state_dict[name] = value

    return distributed_state_dict


def shard_ltx_model(model: LTXModel, model_state_dict: dict, *, cpu_offload: bool = False) -> LTXModel:
    del cpu_offload
    velocity_model = model
    mesh = ulysses_device_mesh()
    if mesh is None:
        raise RuntimeError("FSDP sharding requires an initialized Ulysses device mesh")

    ignored_params = {
        param
        for name, param in velocity_model.named_parameters()
        if not name.startswith("transformer_blocks.")
    }

    mp_policy = MixedPrecisionPolicy(param_dtype=None, reduce_dtype=None, output_dtype=None, cast_forward_inputs=True)

    for index, block in enumerate(velocity_model.transformer_blocks):
        velocity_model.transformer_blocks[index] = fully_shard(
            module=block,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
        )

    fully_shard(
        velocity_model,
        mesh=mesh,
        ignored_params=ignored_params,
        mp_policy=mp_policy,
        reshard_after_forward=True,
    )

    distributed_state_dict = _distribute_full_state_dict(velocity_model, model_state_dict)
    incompatible = velocity_model.load_state_dict(distributed_state_dict, strict=False, assign=True)
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected keys while loading FSDP LTX model: {incompatible.unexpected_keys}")
    if incompatible.missing_keys:
        raise RuntimeError(f"Missing keys while loading FSDP LTX model: {incompatible.missing_keys}")
    return velocity_model
