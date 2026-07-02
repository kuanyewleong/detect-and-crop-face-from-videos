python crop_faces_from_videos.py \
  --video_dir sample_videos \
  --output_dir faces_from_comfyui \
  --num_faces 8 \
  --attempts_per_segment 12 \
  --min_confidence 0.90 \
  --fallback_min_confidence 0.80 \
  --random_fill_batch_size 32 \
  --max_random_fill_attempts 3000 \
  --resize 512