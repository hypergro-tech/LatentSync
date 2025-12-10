
#!/usr/bin/env python3

import argparse
import os
import subprocess
import urllib.request
import urllib.error
from datetime import datetime
import logging
import sys
import shutil

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_file(url, temp_dir):
    """Download file from URL to temporary directory with proper error handling"""
    try:
        print(f"Starting download from: {url}")

        # Create a request with headers to avoid blocking
        request = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })

        # Get filename from URL
        filename = os.path.basename(url.split('?')[0])  # Remove query parameters
        if not filename or '.' not in filename:
            # Try to get filename from response headers
            try:
                response = urllib.request.urlopen(request)
                content_disposition = response.headers.get('content-disposition')
                if content_disposition:
                    import re
                    filename_match = re.findall('filename=(.+)', content_disposition)
                    if filename_match:
                        filename = filename_match[0].strip('"')
                response.close()
            except:
                pass

            # Fall back to generic name with extension guessing
            if not filename or '.' not in filename:
                if 'video' in url.lower() or url.endswith('.mp4'):
                    filename = "downloaded_video.mp4"
                elif 'audio' in url.lower() or url.endswith(('.mp3', '.wav', '.mpga')):
                    filename = "downloaded_audio" + (os.path.splitext(url)[1] or '.mp3')
                else:
                    filename = "downloaded_file"

        filepath = os.path.join(temp_dir, filename)
        filepath = os.path.abspath(filepath)

        print(f"Downloading to: {filepath}")
        print(f"File will be saved as: {filename}")

        # Download with progress
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size * 100) / total_size)
                print(f"\rDownload progress: {percent:.1f}%", end='', flush=True)

        urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
        print()  # New line after progress

        # Verify download
        if not os.path.exists(filepath):
            raise RuntimeError(f"Download failed - file not created: {filepath}")

        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise RuntimeError(f"Download failed - empty file: {filepath}")

        print(f"Download successful: {filepath} ({file_size} bytes)")
        return filepath

    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP Error {e.code}: {e.reason} for URL: {url}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL Error: {e.reason} for URL: {url}")
    except Exception as e:
        raise RuntimeError(f"Download failed for {url}: {str(e)}")


def run_ffmpeg_command(cmd, description="FFmpeg operation"):
    """Run FFmpeg command with error handling"""
    try:
        print(f"Running {description}...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"{description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"{description} failed:")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error: {e.stderr}")
        raise RuntimeError(f"{description} failed: {e.stderr}")
    except FileNotFoundError:
        raise RuntimeError("FFmpeg not found. Please install FFmpeg and ensure it's in your PATH")


def convert_audio_to_wav(input_path, output_path):
    """Convert audio file to WAV format using ffmpeg"""
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)

    cmd = [
        'ffmpeg', '-y', '-i', input_path,
        '-ar', '16000',  # Sample rate for whisper
        '-ac', '1',      # Mono channel
        '-c:a', 'pcm_s16le',  # PCM 16-bit for compatibility
        '-f', 'wav',
        output_path
    ]

    run_ffmpeg_command(cmd, f"Converting audio to WAV: {os.path.basename(input_path)}")
    return output_path


def get_media_info(media_path):
    """Get media duration and other info using ffprobe"""
    try:
        # Get duration
        duration_cmd = [
            'ffprobe', '-v', 'quiet', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', media_path
        ]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True, check=True)
        duration = float(duration_result.stdout.strip())

        # Check for audio stream
        audio_cmd = [
            'ffprobe', '-v', 'quiet', '-select_streams', 'a',
            '-show_entries', 'stream=index', '-of', 'csv=p=0', media_path
        ]
        audio_result = subprocess.run(audio_cmd, capture_output=True, text=True)
        has_audio = bool(audio_result.stdout.strip())

        return {
            'duration': duration,
            'has_audio': has_audio,
            'path': media_path
        }

    except (subprocess.CalledProcessError, ValueError) as e:
        logger.warning(f"Could not get info for {media_path}: {e}")
        return {'duration': 0, 'has_audio': False, 'path': media_path}


def create_looped_video(video_path, target_duration, temp_dir):
    """Create a looped version of video to match target duration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{temp_dir}/looped_video_{timestamp}.mp4"

    cmd = [
        'ffmpeg', '-y',
        '-stream_loop', '-1',  # Loop indefinitely
        '-i', video_path,
        '-t', str(target_duration),  # Stop at target duration
        '-c:v', 'libx264',
        '-preset', 'fast',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        output_path
    ]

    run_ffmpeg_command(cmd, f"Creating looped video for {target_duration:.2f}s")
    return output_path


def add_silence_to_audio(audio_path, target_duration, temp_dir, location='end'):
    """Add silence to audio to match target duration"""
    audio_info = get_media_info(audio_path)
    current_duration = audio_info['duration']

    if current_duration >= target_duration:
        return audio_path

    silence_duration = target_duration - current_duration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{temp_dir}/padded_audio_{timestamp}.wav"

    if location == 'start':
        # Add silence at start
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'anullsrc=duration={silence_duration}:sample_rate=16000:channel_layout=mono',
            '-i', audio_path,
            '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1[out]',
            '-map', '[out]',
            output_path
        ]
    else:
        # Add silence at end
        cmd = [
            'ffmpeg', '-y',
            '-i', audio_path,
            '-f', 'lavfi', '-i', f'anullsrc=duration={silence_duration}:sample_rate=16000:channel_layout=mono',
            '-filter_complex', '[0:a][1:a]concat=n=2:v=0:a=1[out]',
            '-map', '[out]',
            output_path
        ]

    run_ffmpeg_command(cmd, f"Adding {silence_duration:.2f}s silence to audio ({location})")
    return output_path


def trim_media(media_path, target_duration, temp_dir, media_type='video'):
    """Trim media to target duration"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    extension = '.mp4' if media_type == 'video' else '.wav'
    output_path = f"{temp_dir}/trimmed_{media_type}_{timestamp}{extension}"

    cmd = [
        'ffmpeg', '-y',
        '-i', media_path,
        '-t', str(target_duration),
        '-c', 'copy' if media_type == 'video' else 'pcm_s16le',
        output_path
    ]

    run_ffmpeg_command(cmd, f"Trimming {media_type} to {target_duration:.2f}s")
    return output_path


def pad_audio_for_inference(audio_path, temp_dir, target_fps=25):
    """Pad audio duration to be multiple of 16 frames for model compatibility"""
    audio_info = get_media_info(audio_path)
    duration = audio_info['duration']

    # Calculate frames and check if padding needed
    num_frames = int(duration * target_fps)
    remainder = num_frames % 16

    if remainder == 0:
        print(f"Audio already has {num_frames} frames (multiple of 16)")
        return audio_path, num_frames

    # Calculate required padding
    pad_frames = 16 - remainder
    pad_duration = pad_frames / target_fps
    new_duration = duration + pad_duration

    print(f"Padding audio: {num_frames} -> {num_frames + pad_frames} frames (+{pad_duration:.3f}s)")

    padded_audio = add_silence_to_audio(audio_path, new_duration, temp_dir, 'end')
    final_frames = num_frames + pad_frames

    return padded_audio, final_frames


def match_video_audio_duration(video_path, audio_path, temp_dir, strategy='extend_video'):
    """Match video and audio durations using specified strategy"""
    video_info = get_media_info(video_path)
    audio_info = get_media_info(audio_path)

    video_duration = video_info['duration']
    audio_duration = audio_info['duration']

    print(f"Video duration: {video_duration:.2f}s")
    print(f"Audio duration: {audio_duration:.2f}s")
    print(f"Difference: {abs(video_duration - audio_duration):.2f}s")

    # If durations are very close, don't modify
    if abs(video_duration - audio_duration) < 0.1:
        print("Durations are already very close, no modification needed")
        return video_path, audio_path

    if strategy == 'extend_video' and audio_duration > video_duration:
        print(f"Extending video to match audio duration ({audio_duration:.2f}s)")
        processed_video = create_looped_video(video_path, audio_duration, temp_dir)
        return processed_video, audio_path

    elif strategy == 'add_silence' and video_duration > audio_duration:
        print(f"Adding silence to audio to match video duration ({video_duration:.2f}s)")
        processed_audio = add_silence_to_audio(audio_path, video_duration, temp_dir, 'end')
        return video_path, processed_audio

    elif strategy == 'trim':
        # Trim both to shorter duration
        target_duration = min(video_duration, audio_duration)
        print(f"Trimming both to {target_duration:.2f}s")

        processed_video = video_path
        processed_audio = audio_path

        if video_duration > target_duration:
            processed_video = trim_media(video_path, target_duration, temp_dir, 'video')

        if audio_duration > target_duration:
            processed_audio = trim_media(audio_path, target_duration, temp_dir, 'audio')

        return processed_video, processed_audio

    else:
        print("No duration matching applied with current strategy")
        return video_path, audio_path


def convert_video_fps(video_path, target_fps, temp_dir):
    """Convert video to target FPS"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"{temp_dir}/fps_converted_{target_fps}_{timestamp}.mp4"

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-filter:v', f'fps={target_fps}',
        '-c:v', 'libx264',
        '-preset', 'medium',
        '-crf', '23',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]

    run_ffmpeg_command(cmd, f"Converting video to {target_fps} FPS")
    return output_path


def run_latentsync_inference(video_path, audio_path, temp_dir):
    """Run the actual LatentSync inference"""
    try:
        print("Attempting to load LatentSync models...")

        # Try to import required modules
        from pathlib import Path
        from omegaconf import OmegaConf
        import torch
        from diffusers import AutoencoderKL, DDIMScheduler
        from latentsync.models.unet import UNet3DConditionModel
        from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
        from accelerate.utils import set_seed
        from latentsync.whisper.audio2feature import Audio2Feature

        print("All imports successful!")

        # Convert paths to absolute
        video_path = os.path.abspath(video_path)
        audio_path = os.path.abspath(audio_path)

        # Hardcoded configuration paths
        unet_config_path = "configs/unet/stage2_512.yaml"
        inference_ckpt_path = "checkpoints/latentsync_unet.pt"

        if not os.path.exists(unet_config_path):
            raise FileNotFoundError(f"Config file not found: {unet_config_path}")
        if not os.path.exists(inference_ckpt_path):
            raise FileNotFoundError(f"Checkpoint file not found: {inference_ckpt_path}")

        # Load configuration
        config = OmegaConf.load(unet_config_path)

        # Check CUDA and determine dtype
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for inference")

        is_fp16_supported = torch.cuda.get_device_capability()[0] >= 7
        dtype = torch.float16 if is_fp16_supported else torch.float32

        print(f"Using dtype: {dtype}")
        print(f"CUDA device: {torch.cuda.get_device_name()}")

        # Load models
        print("Loading scheduler...")
        scheduler = DDIMScheduler.from_pretrained("configs")

        # Determine whisper model based on config
        if config.model.cross_attention_dim == 768:
            whisper_model_path = "checkpoints/whisper/small.pt"
        elif config.model.cross_attention_dim == 384:
            whisper_model_path = "checkpoints/whisper/tiny.pt"
        else:
            raise NotImplementedError("cross_attention_dim must be 768 or 384")

        if not os.path.exists(whisper_model_path):
            raise FileNotFoundError(f"Whisper model not found: {whisper_model_path}")

        print("Loading audio encoder...")
        audio_encoder = Audio2Feature(
            model_path=whisper_model_path,
            device="cuda",
            num_frames=config.data.num_frames,
            audio_feat_length=config.data.audio_feat_length,
        )

        print("Loading VAE...")
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0

        print("Loading UNet...")
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(config.model),
            inference_ckpt_path,
            device="cpu",
        )
        unet = unet.to(dtype=dtype)

        print("Creating pipeline...")
        pipeline = LipsyncPipeline(
            vae=vae,
            audio_encoder=audio_encoder,
            unet=unet,
            scheduler=scheduler,
        ).to("cuda")

        # Set parameters
        inference_steps = 20
        guidance_scale = 1.5
        seed = 1247

        # Create pipeline temp directory
        pipeline_temp_dir = os.path.join(temp_dir, "pipeline_temp")
        os.makedirs(pipeline_temp_dir, exist_ok=True)

        # Set seed
        if seed != -1:
            set_seed(seed)
        else:
            torch.seed()

        print(f"Using seed: {torch.initial_seed()}")

        # Prepare output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_output_path = f"{temp_dir}/inference_output_{timestamp}.mp4"

        print("Running inference...")
        print(f"Input video: {video_path}")
        print(f"Input audio: {audio_path}")
        print(f"Output will be: {temp_output_path}")

        # Run pipeline
        pipeline(
            video_path=video_path,
            audio_path=audio_path,
            video_out_path=temp_output_path,
            num_frames=config.data.num_frames,
            num_inference_steps=inference_steps,
            guidance_scale=guidance_scale,
            weight_dtype=dtype,
            width=config.data.resolution,
            height=config.data.resolution,
            mask_image_path=config.data.mask_image_path,
            temp_dir=pipeline_temp_dir,
        )

        if not os.path.exists(temp_output_path):
            raise RuntimeError(f"Inference failed - output not created: {temp_output_path}")

        print(f"Inference completed successfully: {temp_output_path}")
        return temp_output_path

    except ImportError as e:
        print(f"Import error - environment not properly set up: {e}")
        print("Please ensure all LatentSync dependencies are installed")
        raise
    except Exception as e:
        print(f"Inference failed: {e}")
        raise


def replace_audio_in_video(video_path, audio_path, output_path):
    """Replace audio in video with high-quality original audio"""
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-map', '0:v:0',
        '-map', '1:a:0',
        '-shortest',
        output_path
    ]

    run_ffmpeg_command(cmd, "Replacing audio in final video")


def main():
    parser = argparse.ArgumentParser(description="LatentSync inference with URL inputs and complete processing")
    parser.add_argument("--video_url", type=str, required=True, help="URL of the input video")
    parser.add_argument("--audio_url", type=str, required=True, help="URL of the input audio")
    parser.add_argument("--output_path", type=str, required=True, help="Path for output video")
    parser.add_argument("--temp_dir", type=str, default="temp_processing", help="Temporary directory")
    parser.add_argument("--duration_strategy", type=str, choices=['extend_video', 'add_silence', 'trim'],
                        default='extend_video', help="Strategy for handling duration mismatches")
    parser.add_argument("--output_fps", type=int, default=25, help="Output video FPS")
    parser.add_argument("--cleanup", action="store_true", default=True, help="Clean up temp files after processing")

    args = parser.parse_args()

    # Setup directories
    temp_dir = os.path.abspath(args.temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)

    print(f"Temporary directory: {temp_dir}")
    print(f"Output path: {args.output_path}")
    print(f"Duration strategy: {args.duration_strategy}")
    print(f"Output FPS: {args.output_fps}")

    try:
        # Step 1: Download files
        print("\n" + "="*60)
        print("STEP 1: DOWNLOADING FILES")
        print("="*60)

        print("Downloading video...")
        video_path = download_file(args.video_url, temp_dir)

        print("Downloading audio...")
        original_audio_path = download_file(args.audio_url, temp_dir)

        print(f"\nDownloaded files:")
        print(f"Video: {video_path} ({os.path.getsize(video_path):,} bytes)")
        print(f"Audio: {original_audio_path} ({os.path.getsize(original_audio_path):,} bytes)")

        # Step 2: Convert audio to WAV if needed
        print("\n" + "="*60)
        print("STEP 2: AUDIO CONVERSION")
        print("="*60)

        if original_audio_path.lower().endswith('.wav'):
            print("Audio is already in WAV format")
            audio_wav_path = original_audio_path
        else:
            audio_wav_path = os.path.join(temp_dir, "converted_audio.wav")
            convert_audio_to_wav(original_audio_path, audio_wav_path)

        # Step 3: Convert video to 25fps for processing
        print("\n" + "="*60)
        print("STEP 3: VIDEO PREPROCESSING")
        print("="*60)

        video_25fps = convert_video_fps(video_path, 25, temp_dir)

        # Step 4: Match durations
        print("\n" + "="*60)
        print("STEP 4: DURATION MATCHING")
        print("="*60)

        processed_video, processed_audio = match_video_audio_duration(
            video_25fps, audio_wav_path, temp_dir, args.duration_strategy
        )

        # Step 5: Pad audio for model compatibility
        print("\n" + "="*60)
        print("STEP 5: AUDIO PADDING FOR INFERENCE")
        print("="*60)

        padded_audio, num_frames = pad_audio_for_inference(processed_audio, temp_dir)
        print(f"Audio prepared for inference: {num_frames} frames")

        # Step 6: Run LatentSync inference
        print("\n" + "="*60)
        print("STEP 6: LATENTSYNC INFERENCE")
        print("="*60)

        inference_output = run_latentsync_inference(processed_video, padded_audio, temp_dir)

        # Step 7: Convert to desired output FPS
        print("\n" + "="*60)
        print("STEP 7: OUTPUT FPS CONVERSION")
        print("="*60)

        if args.output_fps != 25:
            final_video = convert_video_fps(inference_output, args.output_fps, temp_dir)
        else:
            final_video = inference_output

        # Step 8: Replace with original high-quality audio
        print("\n" + "="*60)
        print("STEP 8: FINAL AUDIO REPLACEMENT")
        print("="*60)

        replace_audio_in_video(final_video, original_audio_path, args.output_path)

        # Verify output
        if os.path.exists(args.output_path):
            output_size = os.path.getsize(args.output_path)
            print(f"\n✅ SUCCESS! Output created: {args.output_path} ({output_size:,} bytes)")

            # Show output info
            output_info = get_media_info(args.output_path)
            print(f"Output duration: {output_info['duration']:.2f}s")
            print(f"Has audio: {output_info['has_audio']}")
        else:
            raise RuntimeError("Output file was not created")

        # Cleanup
        if args.cleanup:
            print("\n" + "="*60)
            print("CLEANING UP")
            print("="*60)

            try:
                shutil.rmtree(temp_dir)
                print(f"✅ Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"⚠️  Warning: Could not clean up {temp_dir}: {e}")

        print(f"\n🎉 ALL DONE! Final output: {args.output_path}")
        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print(f"Temporary files preserved in: {temp_dir}")

        # Show temp directory contents for debugging
        if os.path.exists(temp_dir):
            print("\nTemporary files:")
            for root, dirs, files in os.walk(temp_dir):
                level = root.replace(temp_dir, '').count(os.sep)
                indent = '  ' * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = '  ' * (level + 1)
                for file in files:
                    try:
                        file_path = os.path.join(root, file)
                        size = os.path.getsize(file_path)
                        print(f"{subindent}{file} ({size:,} bytes)")
                    except:
                        print(f"{subindent}{file} (size unknown)")

        return 1


if __name__ == "__main__":
    exit(main())
