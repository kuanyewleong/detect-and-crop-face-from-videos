# Detect and Crop Face from Videos

Crop temporally diverse face images from videos using a PyTorch-based MTCNN face detector.

## Overview

This project scans a directory of videos, samples frames across each video, detects faces using `facenet-pytorch` MTCNN, and saves cropped face images to an output folder.

It is designed to capture multiple face crops per video with temporal diversity, using a main sampling stage followed by a fallback random sampling stage if needed.

## Features

- Recursively discovers video files in an input directory
- Splits each video into temporal segments and selects candidate frames
- Uses MTCNN face detection for robust face crops
- Saves a configurable number of face crops per video
- Includes a fallback stage to recover missed crops by random sampling
- Optionally resizes face crops and controls JPEG quality

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- PyTorch
- `facenet-pytorch`
- `Pillow`
- `tqdm`

Install dependencies with pip, for example:

```bash
python -m pip install torch torchvision torchaudio
python -m pip install opencv-python facenet-pytorch pillow tqdm
```

> If you have a CUDA-capable GPU, install the matching CUDA version of PyTorch for better performance.

## Usage

Run the main script with the required input and output directories:

```bash
python crop_faces_from_videos.py \
  --video_dir sample_videos \
  --output_dir output_faces \
  --num_faces 8 \
  --attempts_per_segment 12 \
  --min_confidence 0.80 \
  --fallback_min_confidence 0.80 \
  --random_fill_batch_size 32 \
  --max_random_fill_attempts 3000 \
  --resize 512
```

You can also use the provided `run_script.sh` example.

## Command-line Arguments

- `--video_dir`: Input folder containing videos. Required.
- `--output_dir`: Output folder for cropped faces. Required.
- `--num_faces`: Number of face crops to save per video. Default: `8`.
- `--attempts_per_segment`: Candidate frames to try within each temporal segment. Default: `12`.
- `--batch_size`: Batch size for MTCNN detection. Default: `16`.
- `--min_confidence`: Minimum detection confidence for temporal sampling. Default: `0.90`.
- `--fallback_min_confidence`: Minimum detection confidence during random fallback. Default: `0.80`.
- `--random_fill_batch_size`: Number of random frames to test per fallback round. Default: `32`.
- `--max_random_fill_attempts`: Maximum fallback frame attempts per video. Default: `2000`.
- `--margin_ratio`: Extra margin around the detected face box. Default: `0.25`.
- `--resize`: Resize output crop to this square dimension. Use `0` to keep original crop size. Default: `112`.
- `--jpeg_quality`: JPEG quality for saved crops. Default: `95`.
- `--seed`: Random seed. Default: `42`.
- `--device`: Device for PyTorch detector. Choices: `auto`, `cuda`, `cpu`. Default: `auto`.
- `--overwrite`: Overwrite existing output folders.

## Output

Crops are saved under `output_dir`, organized by sanitized video stem names. Each saved file includes:

- video name
- face index
- segment or fallback tag
- frame index
- detection confidence

Example output path:

```text
output_faces/office_video3/office_video3_face_01_segment_01_frame_000123_conf_0.912.jpg
```

## Notes

- The script uses `torch.cuda.is_available()` when `--device auto` is selected.
- If a video already has enough saved crops and `--overwrite` is not set, it will skip reprocessing that video.
- For best results, use clear videos where the subject is visible and well-lit.

