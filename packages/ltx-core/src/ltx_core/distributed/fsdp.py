from __future__ import annotations

import torch
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from ltx_core.distributed.ulysses import ulysses_device_mesh
from ltx_core.model.transformer import LTXModel


def shard_ltx_model(model: LTXModel, model_state_dict: dict, *, cpu_offload: bool = False) -> LTXModel:
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

    set_model_state_dict(
        model=velocity_model,
        model_state_dict=model_state_dict,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=True,
            cpu_offload=cpu_offload,
        ),
    )
    return velocity_model
