import argparse
import logging
import os
import socket
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.distributed import destroy_ulysses, initialize_ulysses, is_primary_rank
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput, default_2_stage_arg_parser, detect_checkpoint_path
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import (
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser, SimpleDenoiser
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec


@dataclass(frozen=True)
class _PipelineInitConfig:
    checkpoint_path: str
    distilled_lora: list[LoraPathStrengthAndSDOps]
    spatial_upsampler_path: str
    gemma_root: str
    loras: list[LoraPathStrengthAndSDOps]
    device: torch.device | None
    num_gpus: int
    quantization_mode: str | None
    torch_compile: bool


def _serialize_quantization_policy(policy: QuantizationPolicy | None) -> str | None:
    if policy is None:
        return None
    if policy == QuantizationPolicy.fp8_cast():
        return "fp8_cast"
    if policy == QuantizationPolicy.fp8_scaled_mm():
        return "fp8_scaled_mm"
    raise ValueError("Unsupported quantization policy for distributed TI2VidTwoStagesPipeline")


def _deserialize_quantization_policy(mode: str | None) -> QuantizationPolicy | None:
    if mode is None:
        return None
    if mode == "fp8_cast":
        return QuantizationPolicy.fp8_cast()
    if mode == "fp8_scaled_mm":
        return QuantizationPolicy.fp8_scaled_mm()
    raise ValueError(f"Unknown quantization mode: {mode}")


def _distributed_env_present() -> bool:
    required = ("MASTER_ADDR", "MASTER_PORT", "WORLD_SIZE", "RANK", "LOCAL_RANK")
    return all(name in os.environ for name in required)


def _move_to_device(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        moved = tuple(_move_to_device(item, device) for item in value)
        if hasattr(value, "_fields"):
            return type(value)(*moved)
        return moved
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if hasattr(value, "__dict__"):
        for key, item in vars(value).items():
            setattr(value, key, _move_to_device(item, device))
        return value
    return value


def _broadcast_from_primary(value: object | None, device: torch.device) -> object:
    if not dist.is_initialized():
        return value

    payload = [_move_to_device(value, torch.device("cpu")) if is_primary_rank() else None]
    dist.broadcast_object_list(payload, src=0)
    return _move_to_device(payload[0], device)


class TI2VidTwoStagesPipeline:
    """
    Two-stage text/image-to-video generation pipeline.
    Stage 1 generates video at half of the target resolution with CFG guidance (assuming
    full model is used), then Stage 2 upsamples by 2x and refines using a distilled
    LoRA for higher quality output. Supports optional image conditioning via the
    images parameter.
    """

    def __init__(
        self,
        checkpoint_path: str,
        distilled_lora: list[LoraPathStrengthAndSDOps],
        spatial_upsampler_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device | None = None,
        num_gpus: int = 1,
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
    ):
        if num_gpus < 1:
            raise ValueError("`num_gpus` must be >= 1")
        self._init_config = _PipelineInitConfig(
            checkpoint_path=checkpoint_path,
            distilled_lora=list(distilled_lora),
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=list(loras),
            device=device,
            num_gpus=num_gpus,
            quantization_mode=_serialize_quantization_policy(quantization),
            torch_compile=torch_compile,
        )
        self.num_gpus = num_gpus
        self._ulysses_initialized = False
        self.device = device or get_device()
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(checkpoint_path, gemma_root, self.dtype, self.device, registry=registry)
        self.image_conditioner = ImageConditioner(checkpoint_path, self.dtype, self.device, registry=registry)
        self.upsampler = VideoUpsampler(
            checkpoint_path, spatial_upsampler_path, self.dtype, self.device, registry=registry
        )
        self.video_decoder = VideoDecoder(checkpoint_path, self.dtype, self.device, registry=registry)
        self.audio_decoder = AudioDecoder(checkpoint_path, self.dtype, self.device, registry=registry)

        self.stage_1 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.stage_2 = DiffusionStage(
            checkpoint_path,
            self.dtype,
            self.device,
            loras=(*tuple(loras), *distilled_lora),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )

    def _ensure_ulysses_initialized(self) -> None:
        if self.num_gpus > 1 and not self._ulysses_initialized:
            initialize_ulysses(self.num_gpus)
            self._ulysses_initialized = True

    @classmethod
    def _from_init_config(cls, config: _PipelineInitConfig) -> "TI2VidTwoStagesPipeline":
        return cls(
            checkpoint_path=config.checkpoint_path,
            distilled_lora=config.distilled_lora,
            spatial_upsampler_path=config.spatial_upsampler_path,
            gemma_root=config.gemma_root,
            loras=config.loras,
            device=config.device,
            num_gpus=config.num_gpus,
            quantization=_deserialize_quantization_policy(config.quantization_mode),
            registry=None,
            torch_compile=config.torch_compile,
        )

    @staticmethod
    def _distributed_child_worker(
        local_rank: int,
        master_addr: str,
        master_port: str,
        init_config: _PipelineInitConfig,
        call_kwargs: dict,
        result_queue: mp.SimpleQueue,
    ) -> None:
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = master_port
        os.environ["WORLD_SIZE"] = str(init_config.num_gpus)
        os.environ["RANK"] = str(local_rank)
        os.environ["LOCAL_RANK"] = str(local_rank)
        torch.cuda.set_device(local_rank)

        try:
            pipeline = TI2VidTwoStagesPipeline._from_init_config(init_config)
            video, audio = pipeline._run_local_call(call_kwargs)
            if local_rank == 0:
                video_chunks = [chunk.detach().cpu().contiguous() for chunk in video]
                payload = {
                    "video_chunks": video_chunks,
                    "audio_waveform": audio.waveform.detach().cpu().contiguous(),
                    "audio_sampling_rate": audio.sampling_rate,
                }
                result_queue.put(payload)
        except Exception as exc:
            result_queue.put({"error": f"rank {local_rank}: {exc}"})
            raise
        finally:
            destroy_ulysses()

    def _run_distributed_call(self, call_kwargs: dict) -> tuple[Iterator[torch.Tensor], Audio]:
        if not torch.cuda.is_available():
            raise RuntimeError("`num_gpus` > 1 requires CUDA")
        if torch.cuda.device_count() < self.num_gpus:
            raise RuntimeError(f"Requested {self.num_gpus} GPUs, but only {torch.cuda.device_count()} are available")

        master_addr = "127.0.0.1"
        master_port = str(_find_free_port())
        ctx = mp.get_context("spawn")
        result_queue: mp.SimpleQueue = ctx.SimpleQueue()
        processes: list[mp.Process] = []

        for local_rank in range(self.num_gpus):
            process = ctx.Process(
                target=TI2VidTwoStagesPipeline._distributed_child_worker,
                args=(
                    local_rank,
                    master_addr,
                    master_port,
                    self._init_config,
                    call_kwargs,
                    result_queue,
                ),
            )
            process.start()
            processes.append(process)

        result = result_queue.get()

        for process in processes:
            process.join()
            if process.exitcode != 0 and "error" not in result:
                raise RuntimeError(f"LTX distributed worker exited with code {process.exitcode}")

        if "error" in result:
            raise RuntimeError(result["error"])

        video_chunks = result["video_chunks"]
        audio = Audio(
            waveform=result["audio_waveform"],
            sampling_rate=result["audio_sampling_rate"],
        )
        return iter(video_chunks), audio

    def _run_local_call(self, call_kwargs: dict) -> tuple[Iterator[torch.Tensor], Audio]:
        self._ensure_ulysses_initialized()
        return self._call_impl(**call_kwargs)

    def _call_impl(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
        max_batch_size: int = 1,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        prompt_contexts = None
        if is_primary_rank():
            prompt_contexts = self.prompt_encoder(
                [prompt, negative_prompt],
                enhance_first_prompt=enhance_prompt,
                enhance_prompt_image=images[0][0] if len(images) > 0 else None,
                enhance_prompt_seed=seed,
                streaming_prefetch_count=streaming_prefetch_count,
            )
        ctx_p, ctx_n = _broadcast_from_primary(prompt_contexts, self.device)
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        stage_1_conditionings = None
        if is_primary_rank():
            stage_1_conditionings = self.image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=images,
                    height=stage_1_output_shape.height,
                    width=stage_1_output_shape.width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                )
            )
        stage_1_conditionings = _broadcast_from_primary(stage_1_conditionings, self.device)

        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        video_state, audio_state = self.stage_1(
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=create_multimodal_guider_factory(
                    params=video_guider_params,
                    negative_context=v_context_n,
                ),
                audio_guider_factory=create_multimodal_guider_factory(
                    params=audio_guider_params,
                    negative_context=a_context_n,
                ),
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=stage_1_output_shape.width,
            height=stage_1_output_shape.height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=v_context_p, conditionings=stage_1_conditionings),
            audio=ModalitySpec(context=a_context_p),
            streaming_prefetch_count=streaming_prefetch_count,
            max_batch_size=max_batch_size,
        )

        upscaled_video_latent = None
        if is_primary_rank():
            upscaled_video_latent = self.upsampler(video_state.latent[:1])
        upscaled_video_latent = _broadcast_from_primary(upscaled_video_latent, self.device)

        distilled_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_conditionings = None
        if is_primary_rank():
            stage_2_conditionings = self.image_conditioner(
                lambda enc: combined_image_conditionings(
                    images=images,
                    height=height,
                    width=width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                )
            )
        stage_2_conditionings = _broadcast_from_primary(stage_2_conditionings, self.device)

        video_state, audio_state = self.stage_2(
            denoiser=SimpleDenoiser(v_context=v_context_p, a_context=a_context_p),
            sigmas=distilled_sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_2_conditionings,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=upscaled_video_latent,
            ),
            audio=ModalitySpec(
                context=a_context_p,
                noise_scale=distilled_sigmas[0].item(),
                initial_latent=audio_state.latent,
            ),
            streaming_prefetch_count=streaming_prefetch_count,
        )

        if not is_primary_rank():
            empty_audio = Audio(
                waveform=torch.empty((1, 1, 0), dtype=self.dtype, device=self.device),
                sampling_rate=1,
            )
            return iter(()), empty_audio

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        return decoded_video, decoded_audio

    @torch.inference_mode()
    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        streaming_prefetch_count: int | None = None,
        max_batch_size: int = 1,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        call_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "seed": seed,
            "height": height,
            "width": width,
            "num_frames": num_frames,
            "frame_rate": frame_rate,
            "num_inference_steps": num_inference_steps,
            "video_guider_params": video_guider_params,
            "audio_guider_params": audio_guider_params,
            "images": images,
            "tiling_config": tiling_config,
            "enhance_prompt": enhance_prompt,
            "streaming_prefetch_count": streaming_prefetch_count,
            "max_batch_size": max_batch_size,
        }
        if self.num_gpus > 1 and not _distributed_env_present():
            return self._run_distributed_call(call_kwargs)

        self._ensure_ulysses_initialized()
        return self._call_impl(**call_kwargs)


def _build_pipeline_from_args(args: argparse.Namespace) -> TI2VidTwoStagesPipeline:
    return TI2VidTwoStagesPipeline(
        checkpoint_path=args.checkpoint_path,
        distilled_lora=args.distilled_lora,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        num_gpus=args.num_gpus,
        quantization=args.quantization,
        torch_compile=args.compile,
    )


def _run_pipeline(args: argparse.Namespace) -> None:
    pipeline = _build_pipeline_from_args(args)
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=MultiModalGuiderParams(
            cfg_scale=args.video_cfg_guidance_scale,
            stg_scale=args.video_stg_guidance_scale,
            rescale_scale=args.video_rescale_scale,
            modality_scale=args.a2v_guidance_scale,
            skip_step=args.video_skip_step,
            stg_blocks=args.video_stg_blocks,
        ),
        audio_guider_params=MultiModalGuiderParams(
            cfg_scale=args.audio_cfg_guidance_scale,
            stg_scale=args.audio_stg_guidance_scale,
            rescale_scale=args.audio_rescale_scale,
            modality_scale=args.v2a_guidance_scale,
            skip_step=args.audio_skip_step,
            stg_blocks=args.audio_stg_blocks,
        ),
        images=args.images,
        tiling_config=tiling_config,
        streaming_prefetch_count=args.streaming_prefetch_count,
        max_batch_size=args.max_batch_size,
    )

    if is_primary_rank():
        encode_video(
            video=video,
            fps=args.frame_rate,
            audio=audio,
            output_path=args.output_path,
            video_chunks_number=video_chunks_number,
        )


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _distributed_worker(local_rank: int, args: argparse.Namespace) -> None:
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    torch.cuda.set_device(local_rank)
    try:
        _run_pipeline(args)
    finally:
        destroy_ulysses()


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_2_stage_arg_parser(params=params)
    args = parser.parse_args()
    if args.num_gpus > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("`--num-gpus` > 1 requires CUDA")
        if torch.cuda.device_count() < args.num_gpus:
            raise RuntimeError(f"Requested {args.num_gpus} GPUs, but only {torch.cuda.device_count()} are available")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_find_free_port()))
        mp.spawn(_distributed_worker, nprocs=args.num_gpus, args=(args,), join=True)
        return

    _run_pipeline(args)


if __name__ == "__main__":
    main()
