"""
Builds the face recognition gallery from the dataset photos.

Usage:
    python enroll_faces.py                       # uses app/assets/dataset/...
    python enroll_faces.py --dataset <folder>    # one sub-folder per person

Re-run this whenever photos are added or a new person folder is created.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.detection.face_identity import build_gallery, find_default_dataset_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Enroll dataset faces for Motion Reader")
    parser.add_argument("--dataset", default=None, help="Dataset folder with one sub-folder per person")
    parser.add_argument("--out", default=None, help="Output gallery path (default: app/assets/models/face_gallery.npz)")
    args = parser.parse_args()

    dataset = args.dataset or find_default_dataset_dir()
    if not dataset or not os.path.isdir(dataset):
        print("Dataset folder not found. Pass --dataset <folder>.")
        return 1
    print(f"Dataset: {dataset}")

    t0 = time.time()

    def progress(done: int, total: int, name: str) -> None:
        if done % 25 == 0 or done == total:
            print(f"  [{done:4d}/{total}] {name}", flush=True)

    summary = build_gallery(dataset, out_path=args.out, progress=progress)
    print(f"\nDone in {time.time() - t0:.1f}s")
    print(json.dumps(summary, indent=2))

    low = [n for n, c in summary["photos_used"].items() if c < 10]
    if low:
        print(f"\nWARNING: few usable face photos for: {', '.join(low)}. Add clearer frontal photos.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
