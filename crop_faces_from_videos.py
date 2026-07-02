import argparse
import random
import shutil
from pathlib import Path
from typing import List, Tuple, Set

import cv2
import torch
from PIL import Image
from tqdm import tqdm
from facenet_pytorch import MTCNN


VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"
}


def sanitize_folder_name(name: str) -> str:
    """
    Make a safe folder name from the video stem.
    """
    safe = "".join(
        c if c.isalnum() or c in ("-", "_", ".") else "_"
        for c in name
    )
    return safe.strip("_")


def list_videos(video_dir: Path) -> List[Path]:
    """
    Recursively list videos from the input directory.
    """
    videos = []

    for path in video_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            videos.append(path)

    return sorted(videos)


def get_video_frame_count(video_path: Path) -> int:
    """
    Get total number of frames in a video.
    """
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return frame_count


def build_temporal_segments(
    frame_count: int,
    num_segments: int
) -> List[Tuple[int, int]]:
    """
    Split a video into roughly equal temporal segments.

    Example:
    frame_count = 160, num_segments = 8

    Returns:
    [
        (0, 19),
        (20, 39),
        ...
        (140, 159)
    ]
    """
    segments = []

    for i in range(num_segments):
        start = int(round(i * frame_count / num_segments))
        end = int(round((i + 1) * frame_count / num_segments)) - 1

        start = max(0, min(start, frame_count - 1))
        end = max(start, min(end, frame_count - 1))

        segments.append((start, end))

    return segments


def sample_candidate_frames_from_segment(
    start: int,
    end: int,
    attempts_per_segment: int,
    rng: random.Random
) -> List[int]:
    """
    Sample candidate frames from one temporal segment.

    The center frame is tried first.
    The remaining candidates are randomly selected from the same segment.
    """
    if end < start:
        return []

    frame_indices = list(range(start, end + 1))

    if not frame_indices:
        return []

    center = (start + end) // 2
    candidates = [center]

    remaining = [idx for idx in frame_indices if idx != center]
    rng.shuffle(remaining)

    candidates.extend(remaining[: max(0, attempts_per_segment - 1)])

    return candidates


def read_frames_at(
    video_path: Path,
    frame_indices: List[int]
) -> List[Tuple[int, Image.Image]]:
    """
    Read selected frames from a video.

    Returns:
        [(frame_idx, PIL_RGB_image), ...]
    """
    frames = []

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return frames

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

        success, frame_bgr = cap.read()

        if not success or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        frames.append((frame_idx, pil_img))

    cap.release()

    return frames


def expand_box(
    box,
    image_width: int,
    image_height: int,
    margin_ratio: float = 0.25,
    square: bool = True
) -> Tuple[int, int, int, int]:
    """
    Expand a detected face bounding box.

    box format:
        [x1, y1, x2, y2]
    """
    x1, y1, x2, y2 = box

    face_w = x2 - x1
    face_h = y2 - y1

    if square:
        side = max(face_w, face_h)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        side = side * (1.0 + margin_ratio)

        x1 = cx - side / 2.0
        y1 = cy - side / 2.0
        x2 = cx + side / 2.0
        y2 = cy + side / 2.0
    else:
        margin_x = face_w * margin_ratio
        margin_y = face_h * margin_ratio

        x1 -= margin_x
        y1 -= margin_y
        x2 += margin_x
        y2 += margin_y

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(image_width, int(round(x2)))
    y2 = min(image_height, int(round(y2)))

    return x1, y1, x2, y2


def choose_best_face(
    boxes,
    probs,
    min_confidence: float
):
    """
    Since each video contains one person, choose the most confident face.

    If multiple faces are detected, this function returns the highest-confidence box.
    """
    if boxes is None or probs is None:
        return None, None

    best_box = None
    best_prob = -1.0

    for box, prob in zip(boxes, probs):
        if prob is None:
            continue

        if prob >= min_confidence and prob > best_prob:
            best_box = box
            best_prob = float(prob)

    return best_box, best_prob


def save_crop(
    image: Image.Image,
    box,
    output_path: Path,
    margin_ratio: float,
    resize: int,
    jpeg_quality: int
):
    """
    Crop face using bounding box, optionally resize, and save.
    """
    w, h = image.size

    x1, y1, x2, y2 = expand_box(
        box,
        image_width=w,
        image_height=h,
        margin_ratio=margin_ratio,
        square=True
    )

    crop = image.crop((x1, y1, x2, y2))

    if resize > 0:
        crop = crop.resize((resize, resize), Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(output_path, quality=jpeg_quality)


def existing_face_count(folder: Path) -> int:
    """
    Count existing image files in an output folder.
    """
    if not folder.exists():
        return 0

    count = 0

    for p in folder.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            count += 1

    return count


def detect_first_valid_face_in_segment(
    video_path: Path,
    frame_candidates: List[int],
    mtcnn: MTCNN,
    batch_size: int,
    min_confidence: float
):
    """
    Try multiple candidate frames from one temporal segment.

    Returns the first frame where a confident face is detected.
    """
    frames = read_frames_at(video_path, frame_candidates)

    if not frames:
        return None

    for start in range(0, len(frames), batch_size):
        batch = frames[start:start + batch_size]

        frame_ids = [item[0] for item in batch]
        images = [item[1] for item in batch]

        try:
            detect_result = mtcnn.detect(images)

            if detect_result is None:
                continue

            # mtcnn.detect may return either (boxes, probs) or (boxes, probs, landmarks)
            if isinstance(detect_result, (tuple, list)) and len(detect_result) >= 2:
                batch_boxes, batch_probs = detect_result[0], detect_result[1]
            else:
                print(
                    f"[WARN] Unexpected detection result from mtcnn.detect for {video_path}: {type(detect_result)}"
                )
                continue
        except Exception as e:
            print(f"[WARN] Detection failed on {video_path}: {e}")
            continue

        for frame_idx, image, boxes, probs in zip(
            frame_ids,
            images,
            batch_boxes,
            batch_probs
        ):
            best_box, best_prob = choose_best_face(
                boxes=boxes,
                probs=probs,
                min_confidence=min_confidence
            )

            if best_box is not None:
                return frame_idx, image, best_box, best_prob

    return None


def sample_random_frame_indices(
    frame_count: int,
    batch_size: int,
    already_used_frames: Set[int],
    rng: random.Random
) -> List[int]:
    """
    Prefer unused frames first.

    Once all frames have been tried, sample with replacement.
    This allows the fallback stage to keep trying until the target number of faces
    is reached or max_random_fill_attempts is exceeded.
    """
    if frame_count <= 0:
        return []

    if len(already_used_frames) < frame_count:
        unused = [
            idx for idx in range(frame_count)
            if idx not in already_used_frames
        ]

        sample_count = min(batch_size, len(unused))

        if sample_count <= 0:
            return []

        return rng.sample(unused, sample_count)

    return [
        rng.randint(0, frame_count - 1)
        for _ in range(batch_size)
    ]


def fallback_random_sampling_until_full(
    video_path: Path,
    output_dir: Path,
    video_name: str,
    mtcnn: MTCNN,
    rng: random.Random,
    already_used_frames: Set[int],
    frame_count: int,
    saved: int,
    num_faces: int,
    batch_size: int,
    random_fill_batch_size: int,
    max_random_fill_attempts: int,
    fallback_min_confidence: float,
    margin_ratio: float,
    resize: int,
    jpeg_quality: int
) -> int:
    """
    Fail-safe stage.

    If the temporal segment stage saves fewer than num_faces, this function
    repeatedly samples random frames from anywhere in the video until the missing
    crops are filled.

    It stops only when:
    1. num_faces crops have been saved, or
    2. max_random_fill_attempts is reached.
    """
    attempts = 0

    while saved < num_faces and attempts < max_random_fill_attempts:
        remaining_attempts = max_random_fill_attempts - attempts
        current_batch_size = min(random_fill_batch_size, remaining_attempts)

        frame_indices = sample_random_frame_indices(
            frame_count=frame_count,
            batch_size=current_batch_size,
            already_used_frames=already_used_frames,
            rng=rng
        )

        if not frame_indices:
            break

        attempts += len(frame_indices)

        for frame_idx in frame_indices:
            already_used_frames.add(frame_idx)

        frames = read_frames_at(video_path, frame_indices)

        if not frames:
            continue

        for start in range(0, len(frames), batch_size):
            if saved >= num_faces:
                break

            batch = frames[start:start + batch_size]

            frame_ids = [item[0] for item in batch]
            images = [item[1] for item in batch]

            try:
                detect_result = mtcnn.detect(images)

                if detect_result is None:
                    continue

                if isinstance(detect_result, (tuple, list)) and len(detect_result) >= 2:
                    batch_boxes, batch_probs = detect_result[0], detect_result[1]
                else:
                    print(
                        f"[WARN] Unexpected detection result from mtcnn.detect for {video_path}: {type(detect_result)}"
                    )
                    continue
            except Exception as e:
                print(f"[WARN] Random fallback detection failed on {video_path}: {e}")
                continue

            for frame_idx, image, boxes, probs in zip(
                frame_ids,
                images,
                batch_boxes,
                batch_probs
            ):
                if saved >= num_faces:
                    break

                best_box, best_prob = choose_best_face(
                    boxes=boxes,
                    probs=probs,
                    min_confidence=fallback_min_confidence
                )

                if best_box is None:
                    continue

                output_filename = (
                    f"{video_name}_face_{saved + 1:02d}"
                    f"_randomfill"
                    f"_frame_{frame_idx:06d}"
                    f"_conf_{best_prob:.3f}.jpg"
                )

                output_path = output_dir / output_filename

                save_crop(
                    image=image,
                    box=best_box,
                    output_path=output_path,
                    margin_ratio=margin_ratio,
                    resize=resize,
                    jpeg_quality=jpeg_quality
                )

                saved += 1

    if saved < num_faces:
        print(
            f"[WARN] Random fallback saved only {saved}/{num_faces} faces "
            f"for {video_path} after {attempts} random frame attempts."
        )

    return saved


def process_one_video(
    video_path: Path,
    output_root: Path,
    mtcnn: MTCNN,
    rng: random.Random,
    num_faces: int,
    attempts_per_segment: int,
    batch_size: int,
    min_confidence: float,
    fallback_min_confidence: float,
    random_fill_batch_size: int,
    max_random_fill_attempts: int,
    margin_ratio: float,
    resize: int,
    jpeg_quality: int,
    overwrite: bool
) -> int:
    """
    Process one video.

    Main stage:
        Split video into num_faces temporal segments and try to save one face
        from each segment.

    Fallback stage:
        If fewer than num_faces are saved, repeatedly sample random frames
        from anywhere in the video until num_faces are saved or the maximum
        fallback attempt limit is reached.
    """
    video_name = sanitize_folder_name(video_path.stem)
    output_dir = output_root / video_name

    if overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    already_saved = existing_face_count(output_dir)

    if already_saved >= num_faces and not overwrite:
        return already_saved

    output_dir.mkdir(parents=True, exist_ok=True)

    frame_count = get_video_frame_count(video_path)

    if frame_count <= 0:
        print(f"[WARN] Could not read frame count: {video_path}")
        return already_saved

    segments = build_temporal_segments(
        frame_count=frame_count,
        num_segments=num_faces
    )

    saved = already_saved
    used_frames = set()

    for segment_id, (seg_start, seg_end) in enumerate(segments, start=1):
        if saved >= num_faces:
            break

        candidate_frames = sample_candidate_frames_from_segment(
            start=seg_start,
            end=seg_end,
            attempts_per_segment=attempts_per_segment,
            rng=rng
        )

        for idx in candidate_frames:
            used_frames.add(idx)

        result = detect_first_valid_face_in_segment(
            video_path=video_path,
            frame_candidates=candidate_frames,
            mtcnn=mtcnn,
            batch_size=batch_size,
            min_confidence=min_confidence
        )

        if result is None:
            continue

        frame_idx, image, best_box, best_prob = result

        output_filename = (
            f"{video_name}_face_{saved + 1:02d}"
            f"_segment_{segment_id:02d}"
            f"_frame_{frame_idx:06d}"
            f"_conf_{best_prob:.3f}.jpg"
        )

        output_path = output_dir / output_filename

        save_crop(
            image=image,
            box=best_box,
            output_path=output_path,
            margin_ratio=margin_ratio,
            resize=resize,
            jpeg_quality=jpeg_quality
        )

        saved += 1

    if saved < num_faces:
        saved = fallback_random_sampling_until_full(
            video_path=video_path,
            output_dir=output_dir,
            video_name=video_name,
            mtcnn=mtcnn,
            rng=rng,
            already_used_frames=used_frames,
            frame_count=frame_count,
            saved=saved,
            num_faces=num_faces,
            batch_size=batch_size,
            random_fill_batch_size=random_fill_batch_size,
            max_random_fill_attempts=max_random_fill_attempts,
            fallback_min_confidence=fallback_min_confidence,
            margin_ratio=margin_ratio,
            resize=resize,
            jpeg_quality=jpeg_quality
        )

    if saved < num_faces:
        print(
            f"[WARN] Only saved {saved}/{num_faces} faces for video: {video_path}"
        )

    return saved


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Crop temporally diverse face images from videos using "
            "a PyTorch MTCNN face detector."
        )
    )

    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Input folder containing videos."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output folder for cropped faces."
    )

    parser.add_argument(
        "--num_faces",
        type=int,
        default=8,
        help="Number of face crops to save per video."
    )

    parser.add_argument(
        "--attempts_per_segment",
        type=int,
        default=12,
        help=(
            "Number of candidate frames to try inside each temporal segment. "
            "Increase this if detection often fails."
        )
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for MTCNN detection."
    )

    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.90,
        help="Minimum detection confidence during temporal segment sampling."
    )

    parser.add_argument(
        "--fallback_min_confidence",
        type=float,
        default=0.80,
        help=(
            "Minimum detection confidence during random fallback filling. "
            "Usually slightly lower than --min_confidence."
        )
    )

    parser.add_argument(
        "--random_fill_batch_size",
        type=int,
        default=32,
        help="Number of random frames to test per fallback round."
    )

    parser.add_argument(
        "--max_random_fill_attempts",
        type=int,
        default=2000,
        help=(
            "Maximum random frame attempts per video during fallback filling. "
            "This prevents infinite loops if no detectable face exists."
        )
    )

    parser.add_argument(
        "--margin_ratio",
        type=float,
        default=0.25,
        help="Extra margin around detected face box."
    )

    parser.add_argument(
        "--resize",
        type=int,
        default=112,
        help=(
            "Resize output face crop to this square size. "
            "Use 0 to keep original crop size."
        )
    )

    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality for saved crops."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed."
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device for PyTorch detector."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output folders."
    )

    args = parser.parse_args()

    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)

    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory does not exist: {video_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=False,
        keep_all=True,
        device=device
    )

    videos = list_videos(video_dir)

    if not videos:
        print(f"No videos found in: {video_dir}")
        return

    print(f"Found {len(videos)} videos.")

    rng = random.Random(args.seed)

    total_saved = 0
    failed_videos = []

    for video_path in tqdm(videos, desc="Processing videos"):
        saved = process_one_video(
            video_path=video_path,
            output_root=output_dir,
            mtcnn=mtcnn,
            rng=rng,
            num_faces=args.num_faces,
            attempts_per_segment=args.attempts_per_segment,
            batch_size=args.batch_size,
            min_confidence=args.min_confidence,
            fallback_min_confidence=args.fallback_min_confidence,
            random_fill_batch_size=args.random_fill_batch_size,
            max_random_fill_attempts=args.max_random_fill_attempts,
            margin_ratio=args.margin_ratio,
            resize=args.resize,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite
        )

        total_saved += min(saved, args.num_faces)

        if saved < args.num_faces:
            failed_videos.append((video_path, saved))

    print("\nDone.")
    print(f"Videos processed: {len(videos)}")
    print(f"Target faces per video: {args.num_faces}")
    print(f"Total saved face crops: {total_saved}")

    if failed_videos:
        print("\nVideos with fewer than target crops:")
        for video_path, saved in failed_videos:
            print(f"  {video_path} -> {saved}/{args.num_faces}")
    else:
        print("All videos reached the target number of face crops.")


if __name__ == "__main__":
    main()