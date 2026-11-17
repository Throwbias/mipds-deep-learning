"""Run the ten M12 text prompts with the pinned, pretrained SD-Turbo checkpoint."""

import argparse
from dataclasses import replace
from hashlib import sha256
import importlib.metadata
import json
from pathlib import Path
import platform
import time

from diffusers import StableDiffusionPipeline
import numpy as np
import torch

from mipds_fusion.fusion import FusionOutput
from .bridge import prepare_prompt


MODEL_ID = "stabilityai/sd-turbo"
MODEL_REVISION = "b261bac6fd2cf515557d5d0707481eafa0485ec2"
VERSION = "MIPDS-Generation-SDTurbo-512-text-v1"
REPO = Path(__file__).resolve().parents[1]


def file_digest(path):
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_model(model_dir):
    manifest = json.loads((Path(__file__).parent / "model_manifest.json").read_text())
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != MODEL_REVISION:
        raise ValueError("The model manifest differs from the pinned checkpoint.")
    for entry in manifest["files"]:
        path = model_dir / entry["path"]
        if path.stat().st_size != entry["size"] or file_digest(path) != entry["sha256"]:
            raise ValueError(f"The local model file failed verification: {entry['path']}")
    return manifest


def rejected(action):
    try:
        action()
    except ValueError:
        return True
    return False


def bridge_checks():
    """Use synthetic contract fixtures, without claiming an end-to-end video run."""
    base = FusionOutput(torch.zeros(2, 512), torch.ones(2, 2, dtype=torch.bool),
                        torch.ones(2, dtype=torch.bool), torch.tensor([False, True]))

    def prepare(value=base, **kwargs):
        options = dict(row=1, trainer_text="  An empty exercise recording space.  ", confirmed=True)
        options.update(kwargs)
        return prepare_prompt(value, ["fixture-a", "fixture-b"], **options)

    result = prepare()
    vision_only = replace(base, modality_mask=torch.tensor([[True, False], [True, False]]))
    text_only = replace(base, modality_mask=torch.tensor([[False, True], [False, True]]))
    empty = replace(base, modality_mask=torch.zeros(2, 2, dtype=torch.bool), has_information=torch.zeros(2, dtype=torch.bool))
    return {
        "bridge preserves source ID and truncation": result.clip_id == "fixture-b" and result.source_text_truncated,
        "bridge retains confirmed text without feature-to-token conversion": result.prompt == "An empty exercise recording space.",
        "bridge permits vision-only and text-only sources": prepare(vision_only).vision_present and prepare(text_only).text_present,
        "bridge rejects absent source modalities": rejected(lambda: prepare(empty)),
        "bridge rejects unconfirmed descriptions": rejected(lambda: prepare(confirmed=False)) and rejected(lambda: prepare(confirmed="yes")),
        "bridge rejects empty descriptions and invalid rows": rejected(lambda: prepare(trainer_text=" ")) and rejected(lambda: prepare(row=2)),
        "bridge rejects mismatched versions and widths": rejected(lambda: prepare(replace(base, model_version="wrong")))
        and rejected(lambda: prepare(replace(base, fused_embedding=torch.zeros(2, 256)))),
        "bridge rejects nonfinite features and wrong dtypes": rejected(lambda: prepare(replace(base, fused_embedding=torch.full((2, 512), float("nan")))))
        and rejected(lambda: prepare(replace(base, fused_embedding=torch.zeros(2, 512, dtype=torch.float64)))),
        "bridge rejects inconsistent availability and malformed flags": rejected(lambda: prepare(replace(base, has_information=torch.zeros(2, dtype=torch.bool))))
        and rejected(lambda: prepare(replace(base, modality_mask=torch.ones(2, 2, dtype=torch.int32)))),
        "bridge rejects nonzero absent feature blocks": rejected(lambda: prepare(replace(vision_only, fused_embedding=torch.ones(2, 512)))),
    }


def run(model_dir, prompts_path, output_dir, *, device="cpu", dtype_name="bfloat16", threads=4):
    torch.set_num_threads(threads)
    manifest = verify_model(model_dir)
    prompts = json.loads(prompts_path.read_text())
    if len(prompts) != 10 or len({row["id"] for row in prompts}) != 10:
        raise ValueError("The prompting audit requires ten distinct cases.")
    if any(row["id"] != f"p{index:02}" or type(row["seed"]) is not int
           or not isinstance(row["prompt"], str) or not row["prompt"].strip()
           for index, row in enumerate(prompts, 1)):
        raise ValueError("Use p01 through p10, nonempty prompts, and integer seeds.")
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
    started = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_dir, torch_dtype=dtype, variant="fp16", use_safetensors=True,
        local_files_only=True,
    ).to(device)
    for module in (pipe.text_encoder, pipe.unet, pipe.vae):
        module.eval().requires_grad_(False)
    if pipe.scheduler.__class__.__name__ != "EulerDiscreteScheduler":
        raise ValueError("Use the checkpoint's EulerDiscreteScheduler.")
    pipe.set_progress_bar_config(disable=True)
    load_seconds = time.perf_counter() - started
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)
    trace = {}

    def first_tensor(value):
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, "last_hidden_state"):
            return value.last_hidden_state
        if hasattr(value, "sample"):
            return value.sample
        if isinstance(value, (tuple, list)):
            return first_tensor(value[0])
        raise TypeError("The trace expected a tensor output.")

    def capture(name):
        def hook(module, args, output):
            value = first_tensor(output)
            trace[name] = {"shape": list(value.shape), "dtype": str(value.dtype), "finite": bool(torch.isfinite(value).all())}
        return hook

    stages = {
        "clip_embeddings": pipe.text_encoder.text_model.embeddings,
        "clip_final_states": pipe.text_encoder,
        "unet_input_convolution": pipe.unet.conv_in,
        "unet_timestep_embedding": pipe.unet.time_embedding,
        **{f"unet_down_{i}": block for i, block in enumerate(pipe.unet.down_blocks)},
        "unet_mid": pipe.unet.mid_block,
        **{f"unet_up_{i}": block for i, block in enumerate(pipe.unet.up_blocks)},
        "unet_epsilon": pipe.unet,
        "vae_post_quant_convolution": pipe.vae.post_quant_conv,
        "vae_decoder_rgb": pipe.vae.decoder,
    }
    handles = [module.register_forward_hook(capture(name)) for name, module in stages.items()]
    sampler_trace = []

    def after_step(pipeline, step, timestep, values):
        tensor = values["latents"]
        sampler_trace.append({"step": step, "timestep": float(timestep), "shape": list(tensor.shape),
                              "dtype": str(tensor.dtype), "finite": bool(torch.isfinite(tensor).all())})
        return values

    results, first_pixels, token_lengths = [], None, []
    for index, row in enumerate(prompts):
        raw = pipe.tokenizer(row["prompt"], truncation=False)["input_ids"]
        if len(raw) > pipe.tokenizer.model_max_length:
            raise ValueError(f"Prompt {row['id']} would be truncated by the CLIP tokenizer.")
        token_lengths.append(len(raw))
        generator = torch.Generator(device=device).manual_seed(row["seed"])
        before = time.perf_counter()
        with torch.inference_mode():
            image = pipe(
                prompt=row["prompt"], height=512, width=512, num_inference_steps=1,
                guidance_scale=0.0, generator=generator,
                callback_on_step_end=after_step if index == 0 else None,
            ).images[0]
        duration = time.perf_counter() - before
        image_path = image_dir / f"{row['id']}.png"
        image.save(image_path)
        pixels = np.asarray(image)
        if index == 0:
            first_pixels = pixels.copy()
            first_trace = dict(trace)
            for handle in handles:
                handle.remove()
        result = {
            **row, "mode": "standalone_fictional_text", "file": f"images/{row['id']}.png",
            "clip_token_count_with_boundaries": len(raw), "clip_prompt_truncated": False,
            "shape": list(pixels.shape), "image_mode": image.mode,
            "png_sha256": file_digest(image_path), "rgb_sha256": sha256(pixels.tobytes()).hexdigest(),
            "seconds": duration, "pixel_std": float(pixels.std()),
        }
        results.append(result)
        (output_dir / "generation_progress.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"Generated {row['id']} in {duration:.2f}s, {len(raw)} CLIP tokens, PNG {image_path.stat().st_size} bytes.", flush=True)

    with torch.inference_mode():
        repeat = pipe(prompt=prompts[0]["prompt"], height=512, width=512, num_inference_steps=1,
                      guidance_scale=0.0, generator=torch.Generator(device=device).manual_seed(prompts[0]["seed"])).images[0]
    repeat_equal = np.array_equal(first_pixels, np.asarray(repeat))
    checks = bridge_checks()
    checks.update({
        "all pinned model files match the recorded manifest": True,
        "ten distinct prompts produced RGB 512 outputs": len(results) == 10 and all(
            row["shape"] == [512, 512, 3] and row["image_mode"] == "RGB" for row in results
        ),
        "all prompts fit the 77-token CLIP input": max(token_lengths) <= 77,
        "generated outputs are nonconstant": all(row["pixel_std"] > 1 for row in results),
        "same seed reproduces the first RGB output in this runtime": repeat_equal,
        "all observed intermediate tensors are finite": all(stage["finite"] for stage in first_trace.values()),
        "CLIP context has shape 1 by 77 by 1024": first_trace["clip_final_states"]["shape"] == [1, 77, 1024],
        "U-Net predicts four latent channels": first_trace["unet_epsilon"]["shape"] == [1, 4, 64, 64],
        "VAE decodes three image channels": first_trace["vae_decoder_rgb"]["shape"] == [1, 3, 512, 512],
        "one Euler step uses the final training timestep": len(sampler_trace) == 1 and sampler_trace[0]["timestep"] == 999.0,
        "all model parameters remain frozen without gradients": all(not p.requires_grad and p.grad is None
            for module in (pipe.text_encoder, pipe.unet, pipe.vae) for p in module.parameters()),
    })
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        raise AssertionError({name: value for name, value in checks.items() if not value})
    record = {
        "version": VERSION, "model_id": MODEL_ID, "revision": MODEL_REVISION,
        "training_performed": False, "mode": "pretrained_text_to_image_inference",
        "mipds_bridge_evidence": "Synthetic contract fixtures only; examples use the standalone text entry",
        "conditioning": "Checkpoint-native CLIP token states; M10 fused features are not soft prompt tokens",
        "settings": {"height": 512, "width": 512, "steps": 1, "guidance_scale": 0.0,
                     "negative_prompt": None, "scheduler": type(pipe.scheduler).__name__,
                     "device": device, "dtype": dtype_name, "source_weights_variant": "fp16", "threads": threads},
        "software": {name: importlib.metadata.version(name) for name in (
            "torch", "diffusers", "transformers", "accelerate", "huggingface_hub", "safetensors", "numpy", "Pillow"
        )},
        "platform": platform.platform(), "load_seconds": load_seconds,
        "model_manifest_sha256": file_digest(Path(__file__).parent / "model_manifest.json"),
        "prompt_suite_sha256": file_digest(prompts_path), "checks": checks,
        "component_parameter_counts": {name: sum(p.numel() for p in getattr(pipe, name).parameters())
                                       for name in ("text_encoder", "unet", "vae")},
        "tokenizer_max_length": pipe.tokenizer.model_max_length,
        "scheduler_timesteps": pipe.scheduler.timesteps.tolist(), "scheduler_sigmas": pipe.scheduler.sigmas.tolist(),
        "vae_scaling_factor": pipe.vae.config.scaling_factor,
        "upstream_safety_checker": None if pipe.safety_checker is None else type(pipe.safety_checker).__name__,
        "attention_processor": sorted({type(x).__name__ for x in pipe.unet.attn_processors.values()}),
        "first_prompt_trace": first_trace, "sampler_trace": sampler_trace, "results": results,
    }
    (output_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"Completed ten images and {len(checks)} passing checks. Image suitability requires human review.", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, default=REPO / "experiments/m12_diffusion/prompts.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args.model_dir, args.prompts, args.output, device=args.device, dtype_name=args.dtype, threads=args.threads)


if __name__ == "__main__":
    main()
