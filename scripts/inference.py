# At the top of inference.py, make sure you have these imports:
import argparse
import os
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper


def main(config, args):
    if not os.path.exists(args.video_path):
        raise RuntimeError(f"Video path '{args.video_path}' not found")
    if not os.path.exists(args.audio_path):
        raise RuntimeError(f"Audio path '{args.audio_path}' not found")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path: {args.video_path}")
    print(f"Input audio path: {args.audio_path}")
    print(f"Loaded checkpoint path: {args.inference_ckpt_path}")

    # FIXED: Create DDIMScheduler directly (NO HUGGINGFACE)
    scheduler = DDIMScheduler(
        beta_end=0.012,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        clip_sample=False,
        num_train_timesteps=1000,
        prediction_type="epsilon",
        set_alpha_to_one=False,
        steps_offset=1
    )
    print("✓ Created DDIMScheduler directly (fully offline)")

    if config.model.cross_attention_dim == 768:
        whisper_model_name = "small"
    elif config.model.cross_attention_dim == 384:
        whisper_model_name = "tiny"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_name,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    # Load VAE - try HuggingFace first (same as original project), then fallback to local
    vae = None
    try:
        # First try to load from HuggingFace (same as original project)
        print("Attempting to load VAE from HuggingFace: stabilityai/sd-vae-ft-mse")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        print("✓ Successfully loaded VAE from HuggingFace")
    except Exception as e:
        print(f"Failed to load VAE from HuggingFace: {str(e)}")
        print("Trying to load VAE from local checkpoints...")
        
        # Fallback to local loading
        if hasattr(args, 'extension_dir'):
            base_dir = args.extension_dir
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(script_dir)
        
        vae_locations = [
            os.path.join(base_dir, "checkpoints", "vae"),
            os.path.join(base_dir, "checkpoints", "vae", "sd-vae-ft-mse.safetensors"),
            os.path.join(base_dir, "checkpoints", "sd-vae-ft-mse.safetensors"),
            os.path.join(base_dir, "checkpoints", "sd-vae-ft-mse"),
        ]
        
        for vae_path in vae_locations:
            if os.path.exists(vae_path):
                try:
                    if vae_path.endswith('.safetensors'):
                        print(f"Attempting to load VAE from safetensors file: {vae_path}")
                        vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=dtype)
                    elif os.path.isdir(vae_path):
                        print(f"Attempting to load VAE from directory: {vae_path}")
                        vae = AutoencoderKL.from_pretrained(vae_path, torch_dtype=dtype, local_files_only=True)
                    
                    if vae is not None:
                        print(f"✓ Successfully loaded VAE from: {vae_path}")
                        break
                except Exception as e:
                    print(f"Failed to load VAE from {vae_path}: {str(e)}")
                    vae = None
                    continue
        
        if vae is None:
            raise RuntimeError("Failed to load VAE from both HuggingFace and local locations. Please ensure VAE model is available.")

    # Set VAE configuration (same as original project)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    # Rest of the function continues as before...
    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        args.inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # use DeepCache with configurable cache_interval for memory optimization
    cache_interval = getattr(args, 'cache_interval', 3)
    helper = DeepCacheSDHelper(pipe=pipeline)
    helper.set_params(cache_interval=cache_interval, cache_branch_id=0)
    helper.enable()
    print(f"✓ DeepCache enabled with cache_interval={cache_interval}")

    if args.seed != -1:
        set_seed(args.seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    pipeline(
        video_path=args.video_path,
        audio_path=args.audio_path,
        video_out_path=args.video_out_path,
        video_mask_path=args.video_out_path.replace(".mp4", "_mask.mp4"),
        num_frames=config.data.num_frames,
        num_inference_steps=args.inference_steps,
        guidance_scale=args.guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unet_config_path", type=str, default="configs/unet.yaml")
    parser.add_argument("--inference_ckpt_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument("--video_out_path", type=str, required=True)
    parser.add_argument("--inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1247)
    args = parser.parse_args()

    config = OmegaConf.load(args.unet_config_path)

    main(config, args)