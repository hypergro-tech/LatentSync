
#!/usr/bin/env python3

import argparse
import os
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from omegaconf import OmegaConf
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from accelerate.utils import set_seed
from latentsync.whisper.audio2feature import Audio2Feature
from DeepCache import DeepCacheSDHelper


def download_file(url, temp_dir):
    """Download file from URL to temporary directory"""
    filename = os.path.basename(url.split('?')[0])  # Remove query parameters
    if not filename:
        # Generate filename based on content type or use generic name
        filename = "downloaded_file"

    filepath = os.path.join(temp_dir, filename)
    # Convert to absolute path
    filepath = os.path.abspath(filepath)
    print(f"Downloading {url} to {filepath}")
    urllib.request.urlretrieve(url, filepath)
    return filepath


def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg"""
    # Use absolute paths
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    print(f"Converting audio {input_path} to WAV format: {output_path}")

    try:
        subprocess.run([
            'ffmpeg', '-i', input_path,
            '-ar', '16000',  # Sample rate for whisper
            '-ac', '1',      # Mono channel
            '-f', 'wav',
            '-y',            # Overwrite output file
            output_path
        ], check=True, capture_output=True)
        print(f"Audio conversion completed: {output_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg conversion failed: {e.stderr.decode()}")


def replace_audio_in_video(video_path, original_audio_path, output_path):
    """Replace audio in video with original high-quality audio using ffmpeg"""
    # Use absolute paths
    video_path = os.path.abspath(video_path)
    original_audio_path = os.path.abspath(original_audio_path)
    output_path = os.path.abspath(output_path)

    print(f"Replacing audio in {video_path} with original audio from {original_audio_path}")
    print(f"Final output: {output_path}")

    try:
        subprocess.run([
            'ffmpeg',
            '-i', video_path,           # Input video (with lip-sync)
            '-i', original_audio_path,  # Original high-quality audio
            '-c:v', 'copy',             # Copy video stream without re-encoding
            '-c:a', 'aac',              # Encode audio as AAC
            '-b:a', '192k',             # High quality audio bitrate
            '-map', '0:v:0',            # Use video from first input
            '-map', '1:a:0',            # Use audio from second input
            '-shortest',                # Match duration to shortest stream
            '-y',                       # Overwrite output file
            output_path
        ], check=True, capture_output=True)
        print(f"Audio replacement completed: {output_path}")
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"FFmpeg audio replacement failed: {e.stderr.decode()}")


def main_inference(video_path, audio_wav_path, temp_output_path):
    """Main inference method with hardcoded config paths"""

    # Convert all paths to absolute paths
    video_path = os.path.abspath(video_path)
    audio_wav_path = os.path.abspath(audio_wav_path)
    temp_output_path = os.path.abspath(temp_output_path)

    # Hardcoded configuration paths
    unet_config_path = "configs/unet/stage2_512.yaml"
    inference_ckpt_path = "checkpoints/latentsync_unet.pt"

    # Load configuration
    config = OmegaConf.load(unet_config_path)

    # Check input files exist with detailed info
    print(f"Checking video file: {video_path}")
    print(f"Video file exists: {os.path.exists(video_path)}")
    if os.path.exists(video_path):
        print(f"Video file size: {os.path.getsize(video_path)} bytes")
    else:
        raise RuntimeError(f"Video path '{video_path}' not found")

    print(f"Checking audio file: {audio_wav_path}")
    print(f"Audio file exists: {os.path.exists(audio_wav_path)}")
    if os.path.exists(audio_wav_path):
        print(f"Audio file size: {os.path.getsize(audio_wav_path)} bytes")
    else:
        raise RuntimeError(f"Audio path '{audio_wav_path}' not found")

    # Check if the GPU supports float16
    is_fp16_supported = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 7
    dtype = torch.float16 if is_fp16_supported else torch.float32

    print(f"Input video path (absolute): {video_path}")
    print(f"Input audio path (absolute): {audio_wav_path}")
    print(f"Loaded checkpoint path: {inference_ckpt_path}")
    print(f"Temporary output path (absolute): {temp_output_path}")

    scheduler = DDIMScheduler.from_pretrained("configs")

    if config.model.cross_attention_dim == 768:
        whisper_model_path = "checkpoints/whisper/small.pt"
    elif config.model.cross_attention_dim == 384:
        whisper_model_path = "checkpoints/whisper/tiny.pt"
    else:
        raise NotImplementedError("cross_attention_dim must be 768 or 384")

    audio_encoder = Audio2Feature(
        model_path=whisper_model_path,
        device="cuda",
        num_frames=config.data.num_frames,
        audio_feat_length=config.data.audio_feat_length,
    )

    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0

    unet, _ = UNet3DConditionModel.from_pretrained(
        OmegaConf.to_container(config.model),
        inference_ckpt_path,
        device="cpu",
    )

    unet = unet.to(dtype=dtype)

    pipeline = LipsyncPipeline(
        vae=vae,
        audio_encoder=audio_encoder,
        unet=unet,
        scheduler=scheduler,
    ).to("cuda")

    # Default parameters
    inference_steps = 20
    guidance_scale = 1.5
    seed = 1247
    temp_dir = "temp"

    if seed != -1:
        set_seed(seed)
    else:
        torch.seed()

    print(f"Initial seed: {torch.initial_seed()}")

    # Double-check files exist right before pipeline call
    print(f"Final check - Video exists: {os.path.exists(video_path)}")
    print(f"Final check - Audio exists: {os.path.exists(audio_wav_path)}")

    # Run the pipeline with absolute paths
    pipeline(
        video_path=video_path,
        audio_path=audio_wav_path,
        video_out_path=temp_output_path,
        num_frames=config.data.num_frames,
        num_inference_steps=inference_steps,
        guidance_scale=guidance_scale,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=temp_dir,
    )

    print(f"Lip-sync inference completed. Temporary output saved to: {temp_output_path}")


def main():
    parser = argparse.ArgumentParser(description="LatentSync inference with URL inputs")
    parser.add_argument("--video_url", type=str, required=True, help="URL of the input video")
    parser.add_argument("--audio_url", type=str, required=True, help="URL of the input audio")
    parser.add_argument("--output_path", type=str, required=True, help="Path for output video")
    parser.add_argument("--temp_dir", type=str, default="temp", help="Temporary directory for downloads")

    args = parser.parse_args()

    # Create temporary directory with absolute path
    temp_dir = os.path.abspath(args.temp_dir)
    os.makedirs(temp_dir, exist_ok=True)
    print(f"Using temporary directory: {temp_dir}")

    video_path = None
    original_audio_path = None
    audio_wav_path = None
    temp_output_path = None

    try:
        # Download video and audio files
        print("Downloading video and audio files...")
        video_path = download_file(args.video_url, temp_dir)
        original_audio_path = download_file(args.audio_url, temp_dir)

        # Verify files were downloaded successfully
        if not os.path.exists(video_path):
            raise RuntimeError(f"Failed to download video from {args.video_url}")
        if not os.path.exists(original_audio_path):
            raise RuntimeError(f"Failed to download audio from {args.audio_url}")

        print(f"Video downloaded: {video_path} (size: {os.path.getsize(video_path)} bytes)")
        print(f"Audio downloaded: {original_audio_path} (size: {os.path.getsize(original_audio_path)} bytes)")

        # Convert audio to WAV for inference if needed
        audio_wav_path = os.path.join(temp_dir, "audio_for_inference.wav")
        audio_wav_path = os.path.abspath(audio_wav_path)

        # Check if audio is already in WAV format
        if original_audio_path.lower().endswith('.wav'):
            print("Audio is already in WAV format")
            audio_wav_path = original_audio_path
        else:
            convert_audio_to_wav(original_audio_path, audio_wav_path)

        # Create temporary output path for lip-sync video (without final audio)
        temp_output_path = os.path.join(temp_dir, "lipsync_output_temp.mp4")
        temp_output_path = os.path.abspath(temp_output_path)

        # Run main inference
        print("Running lip-sync inference...")
        main_inference(video_path, audio_wav_path, temp_output_path)

        # Replace audio with original high-quality audio
        print("Replacing with original high-quality audio...")
        output_path = os.path.abspath(args.output_path)
        replace_audio_in_video(temp_output_path, original_audio_path, output_path)

        print(f"Process completed successfully! Final output: {output_path}")

        # Only cleanup on success
        print("Cleaning up temporary files...")
        cleanup_files = [video_path, original_audio_path]
        if audio_wav_path != original_audio_path:  # Don't delete twice if same file
            cleanup_files.append(audio_wav_path)
        if temp_output_path and os.path.exists(temp_output_path):
            cleanup_files.append(temp_output_path)

        for file_path in cleanup_files:
            try:
                if file_path and os.path.exists(file_path):
                    os.unlink(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Warning: Could not delete {file_path}: {e}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        print(f"Temporary files preserved for debugging in: {temp_dir}")

        # Print debug info about existing files
        if os.path.exists(temp_dir):
            print("Files in temp directory:")
            for f in os.listdir(temp_dir):
                f_path = os.path.join(temp_dir, f)
                if os.path.isfile(f_path):
                    print(f"  - {f} ({os.path.getsize(f_path)} bytes)")

        return 1


if __name__ == "__main__":
    exit(main())
