"""
Captures face photos of a new person from the webcam into the dataset.

Usage:
    python capture_person.py --name Kyle            # ~150 photos, camera 0
    python capture_person.py --name Kyle --count 200 --camera 1

A preview window guides the person through several head angles. A photo is
saved only when one clear face is visible, so the shots are usable for
enrollment. Press Q or Esc to stop early. Afterwards run enroll_faces.py
(or pass --enroll) so the app recognizes the new person.
"""
import argparse
import os
import sys
import time
from datetime import datetime

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.detection.face_identity import FaceIdentifier, find_default_dataset_dir

# (instruction, share of the photos)
PHASES = [
    ("Look straight at the camera", 0.25),
    ("Slowly turn your head LEFT", 0.15),
    ("Slowly turn your head RIGHT", 0.15),
    ("Tilt your head UP a little", 0.10),
    ("Tilt your head DOWN a little", 0.10),
    ("Move a bit CLOSER, then FARTHER", 0.15),
    ("Smile / change expression", 0.10),
]
MIN_INTERVAL = 0.30          # seconds between saved photos
MIN_FACE_PX = 90


def open_camera(index: int, width: int, height: int):
    for api in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, api)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            ok, frame = cap.read()
            if ok and frame is not None:
                return cap
        cap.release()
    return None


def put_text(img, text, org, scale=0.8, color=(255, 255, 255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a new person's photos for Motion Reader")
    parser.add_argument("--name", required=True, help="Person's name (becomes the dataset folder name)")
    parser.add_argument("--count", type=int, default=150, help="Number of photos to capture")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument("--dataset", default=None, help="Dataset folder (default: the app's dataset)")
    parser.add_argument("--enroll", action="store_true", help="Rebuild the face gallery afterwards")
    args = parser.parse_args()

    name = args.name.strip()
    if not name or any(c in name for c in '\\/:*?"<>|'):
        print("Invalid name.")
        return 1
    dataset = args.dataset or find_default_dataset_dir()
    if not dataset:
        print("Dataset folder not found. Pass --dataset <folder>.")
        return 1
    out_dir = os.path.join(dataset, name)
    os.makedirs(out_dir, exist_ok=True)
    existing = len([f for f in os.listdir(out_dir) if f.lower().endswith(".jpg")])

    detector = FaceIdentifier(gallery_path="__none__")
    if not detector.models_loaded:
        print(detector.error)
        return 1

    cap = open_camera(args.camera, 1280, 720)
    if cap is None:
        print(f"Cannot open camera {args.camera}. Close any app using it (e.g. Motion Reader) and retry.")
        return 1

    window = f"Capturing {name} - press Q to stop"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1100, 640)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    targets = []
    acc = 0
    for i, (_, share) in enumerate(PHASES):
        n = args.count - acc if i == len(PHASES) - 1 else int(round(args.count * share))
        targets.append(n)
        acc += n

    saved, phase, phase_saved = 0, 0, 0
    last_save = 0.0
    start = time.time()
    countdown_end = start + 3.0
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Saving to {out_dir} ({existing} photos already there)")

    while saved < args.count:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.02)
            continue
        now = time.time()
        view = cv2.flip(frame, 1)  # mirror preview so moving left looks natural

        faces = detector.detect(frame)
        good = [f for f in faces if f[2] >= MIN_FACE_PX and f[14] >= 0.85]
        status, color = "", (0, 200, 255)
        if len(faces) == 0:
            status, color = "No face found - face the camera", (0, 0, 255)
        elif len(good) == 0:
            status, color = "Come closer / improve lighting", (0, 165, 255)
        elif len(good) > 1:
            status, color = "Only one person in view, please", (0, 0, 255)

        for f in faces:
            x, y, w, h = (int(v) for v in f[:4])
            mx = frame.shape[1] - (x + w)  # mirrored x for preview
            col = (0, 220, 0) if (len(good) == 1 and f is good[0]) else (0, 0, 255)
            cv2.rectangle(view, (mx, y), (mx + w, y + h), col, 2)

        if now < countdown_end:
            put_text(view, f"Get ready... {int(countdown_end - now) + 1}", (40, 120), 1.6, (0, 255, 255), 3)
        elif len(good) == 1 and now - last_save >= MIN_INTERVAL:
            fname = os.path.join(out_dir, f"{name}_{stamp}_{existing + saved + 1:03d}.jpg")
            if cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                saved += 1
                phase_saved += 1
                last_save = now
                if phase_saved >= targets[phase] and phase < len(PHASES) - 1:
                    phase += 1
                    phase_saved = 0
            status, color = "Capturing...", (0, 220, 0)

        # HUD
        put_text(view, PHASES[phase][0], (40, 60), 1.1, (255, 255, 0), 3)
        if status:
            put_text(view, status, (40, view.shape[0] - 70), 0.9, color, 2)
        bar_w = int((view.shape[1] - 80) * saved / max(1, args.count))
        cv2.rectangle(view, (40, view.shape[0] - 40), (view.shape[1] - 40, view.shape[0] - 20), (60, 60, 60), -1)
        cv2.rectangle(view, (40, view.shape[0] - 40), (40 + bar_w, view.shape[0] - 20), (0, 220, 0), -1)
        put_text(view, f"{saved}/{args.count}", (view.shape[1] - 200, 60), 1.0, (255, 255, 255), 2)
        cv2.imshow(window, view)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break
        if now - start > 300:
            print("Timed out after 5 minutes.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captured {saved} photos of {name} in {time.time() - start:.0f}s -> {out_dir}")

    if args.enroll and saved > 0:
        import subprocess
        return subprocess.call([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "enroll_faces.py"),
                                "--dataset", dataset])
    return 0 if saved > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
