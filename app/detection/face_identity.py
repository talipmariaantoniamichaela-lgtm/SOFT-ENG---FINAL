"""
Face Identity Recognition for Motion Reader.

Recognizes the enrolled people from the dataset using pretrained face models
shipped with OpenCV (no retraining needed):
  - YuNet  (face_detection_yunet_2023mar.onnx)  -> face boxes + 5 landmarks
  - SFace  (face_recognition_sface_2021dec.onnx) -> 128-d identity embedding

Enrollment ("gallery") embeds every usable face photo in the dataset once and
stores the embeddings in face_gallery.npz. At runtime each detected face is
embedded and compared against the gallery by cosine similarity; faces that do
not match anyone closely enough are reported as unknown.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


YUNET_FILE = "face_detection_yunet_2023mar.onnx"
SFACE_FILE = "face_recognition_sface_2021dec.onnx"
GALLERY_FILE = "face_gallery.npz"

# A face is accepted as an enrolled person only if its top-k similarity to that
# person reaches THRESHOLD and beats the runner-up person by MARGIN. Dataset
# photos are all from one outdoor session, so dataset-calibrated absolute
# thresholds come out far too strict for a webcam; the margin rule is robust to
# the overall drop in similarity under different lighting / cameras.
DEFAULT_THRESHOLD = 0.45
DEFAULT_MARGIN = 0.15
TOP_K = 5
# Faces whose calibrated match probability reaches this are given the name
ACCEPT_PROBABILITY = 0.5
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def default_models_dir() -> str:
    if hasattr(sys, "_MEIPASS"):
        candidate = os.path.join(sys._MEIPASS, "app", "assets", "models")
        if os.path.isdir(candidate):
            return candidate
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "models")


@dataclass
class FaceMatch:
    box: Tuple[int, int, int, int]      # (x, y, w, h) in frame pixels
    name: Optional[str]                 # None when the face matches nobody
    score: float                        # cosine similarity to the best person
    best_guess: str                     # closest enrolled person, even if below threshold
    det_score: float
    probability: float = 0.0            # calibrated chance (0-1) that the face is best_guess


class FaceIdentifier:
    """Detects faces in a frame and matches them against the enrolled gallery."""

    def __init__(
        self,
        models_dir: Optional[str] = None,
        gallery_path: Optional[str] = None,
        detection_width: int = 960,
        threshold: Optional[float] = None,
        margin: Optional[float] = None,
    ):
        self.models_dir = models_dir or default_models_dir()
        self.gallery_path = gallery_path or os.path.join(self.models_dir, GALLERY_FILE)
        self.detection_width = detection_width
        self._threshold_override = threshold
        self._margin_override = margin

        self.detector = None
        self.recognizer = None
        self.names: List[str] = []
        self.gallery: Optional[np.ndarray] = None      # (M, 128) L2-normalized
        self.labels: Optional[np.ndarray] = None       # (M,) index into self.names
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD
        self.margin = margin if margin is not None else DEFAULT_MARGIN
        # Logistic calibration: P(match) = sigmoid(w_score*score + w_margin*margin + bias)
        self.calibration: Optional[Tuple[float, float, float]] = None
        self.error: Optional[str] = None

        self._load_models()
        if self.models_loaded and os.path.exists(self.gallery_path):
            self.load_gallery(self.gallery_path)

    # ------------------------------------------------------------------
    # Model / gallery loading
    # ------------------------------------------------------------------
    def _load_models(self) -> None:
        det_path = os.path.join(self.models_dir, YUNET_FILE)
        rec_path = os.path.join(self.models_dir, SFACE_FILE)
        missing = [p for p in (det_path, rec_path) if not os.path.exists(p)]
        if missing:
            self.error = "Missing face model(s): " + ", ".join(os.path.basename(p) for p in missing)
            return
        try:
            self.detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320), 0.75, 0.3, 50)
            self.recognizer = cv2.FaceRecognizerSF.create(rec_path, "")
        except Exception as e:  # pragma: no cover - depends on OpenCV build
            self.error = f"Failed to load face models: {e}"
            self.detector = None
            self.recognizer = None

    @property
    def models_loaded(self) -> bool:
        return self.detector is not None and self.recognizer is not None

    @property
    def is_ready(self) -> bool:
        return self.models_loaded and self.gallery is not None and len(self.names) > 0

    def load_gallery(self, path: str) -> bool:
        try:
            data = np.load(path, allow_pickle=False)
            self.gallery = data["embeddings"].astype(np.float32)
            self.labels = data["labels"].astype(np.int32)
            self.names = [str(n) for n in data["names"]]
            if self._threshold_override is None and "threshold" in data:
                self.threshold = float(data["threshold"])
            if self._margin_override is None and "margin" in data:
                self.margin = float(data["margin"])
            self.calibration = tuple(float(v) for v in data["calibration"]) if "calibration" in data else None
            self.gallery_path = path
            return True
        except Exception as e:
            self.error = f"Failed to load face gallery: {e}"
            self.gallery = None
            self.labels = None
            self.names = []
            return False

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def detect(self, bgr: np.ndarray, max_width: Optional[int] = None) -> np.ndarray:
        """Returns YuNet face rows (N x 15) in the coordinates of `bgr`."""
        if self.detector is None or bgr is None or bgr.size == 0:
            return np.zeros((0, 15), dtype=np.float32)
        h, w = bgr.shape[:2]
        max_width = max_width or self.detection_width
        scale = min(1.0, max_width / float(w))
        img = cv2.resize(bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else bgr
        self.detector.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.detector.detect(img)
        if faces is None:
            return np.zeros((0, 15), dtype=np.float32)
        faces = faces.copy()
        if scale < 1.0:
            faces[:, :14] /= scale
        return faces

    def embed(self, bgr: np.ndarray, face_row: np.ndarray) -> np.ndarray:
        """Aligns the face and returns its L2-normalized 128-d embedding."""
        aligned = self.recognizer.alignCrop(bgr, face_row)
        return self.embed_aligned(aligned)

    def embed_aligned(self, aligned: np.ndarray) -> np.ndarray:
        feat = self.recognizer.feature(aligned).reshape(-1).astype(np.float32)
        return feat / (np.linalg.norm(feat) + 1e-9)

    def probability(self, score: float, margin: float) -> float:
        """
        Chance (0-1) that a face with this top-k similarity and lead over the
        runner-up really is that enrolled person. Calibrated by enroll_faces.py on
        webcam-quality copies of the dataset photos.
        """
        if self.calibration is None:
            # Uncalibrated gallery: a rough ramp around the accept rule
            return float(np.clip(0.5 + (score - self.threshold) * 1.5 + (margin - self.margin) * 2.0, 0.0, 1.0))
        w_s, w_m, b = self.calibration
        return float(1.0 / (1.0 + np.exp(-(w_s * score + w_m * margin + b))))

    def is_accepted(self, score: float, margin: float, probability: float) -> bool:
        if self.calibration is not None and self._threshold_override is None and self._margin_override is None:
            return probability >= ACCEPT_PROBABILITY
        return score >= self.threshold and margin >= self.margin

    def match(self, embedding: np.ndarray) -> Tuple[Optional[str], float, str, Dict[str, float], float]:
        """
        Matches one embedding.
        Returns (name or None, score, best_guess, per-person scores, probability of best_guess).
        """
        if not self.is_ready:
            return None, 0.0, "", {}, 0.0
        sims = self.gallery @ embedding
        per_person: Dict[str, float] = {}
        for idx, name in enumerate(self.names):
            s = sims[self.labels == idx]
            if s.size == 0:
                continue
            k = min(TOP_K, s.size)
            per_person[name] = float(np.mean(np.partition(s, -k)[-k:]))
        if not per_person:
            return None, 0.0, "", {}, 0.0
        ranked = sorted(per_person.values(), reverse=True)
        best = max(per_person, key=per_person.get)
        score = per_person[best]
        margin = score - ranked[1] if len(ranked) > 1 else 1.0
        prob = self.probability(score, margin)
        accepted = self.is_accepted(score, margin, prob)
        return (best if accepted else None), score, best, per_person, prob

    def identify(self, bgr: np.ndarray, min_face_px: int = 28) -> List[FaceMatch]:
        """Detects every face in a BGR frame and identifies each one."""
        results: List[FaceMatch] = []
        if not self.models_loaded:
            return results
        for row in self.detect(bgr):
            x, y, w, h = (int(round(v)) for v in row[:4])
            if w < min_face_px or h < min_face_px:
                continue
            name, score, guess, prob = None, 0.0, "", 0.0
            if self.is_ready:
                name, score, guess, _, prob = self.match(self.embed(bgr, row))
            results.append(FaceMatch(box=(x, y, w, h), name=name, score=score, best_guess=guess,
                                     det_score=float(row[14]), probability=prob))
        return results

    # ------------------------------------------------------------------
    # Enrollment
    # ------------------------------------------------------------------
    def enroll_image(self, bgr: np.ndarray, min_face_px: int = 60, min_det_score: float = 0.85,
                     rng: Optional[np.random.Generator] = None) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """
        Embeds the main (largest) face of a dataset photo, plus augmented copies
        that mimic webcam conditions: mirrored, low resolution, and darker.
        Also returns a "probe" embedding of a simulated webcam frame of the photo
        (used only to calibrate match percentages, not added to the gallery).
        Returns ([], None) if the photo has no usable face.
        """
        # Close-up photos have faces too large for YuNet at full size; try a few
        # detection scales (coordinates come back in full resolution either way)
        row = None
        for width in (640, 400, 960):
            faces = self.detect(bgr, max_width=width)
            if len(faces):
                cand = max(faces, key=lambda f: f[2] * f[3])
                if cand[2] >= min_face_px and cand[14] >= min_det_score:
                    row = cand
                    break
        if row is None:
            return [], None
        aligned = self.recognizer.alignCrop(bgr, row)
        variants = [
            aligned,
            cv2.flip(aligned, 1),
            cv2.resize(cv2.resize(aligned, (40, 40), interpolation=cv2.INTER_AREA), (112, 112)),
            cv2.convertScaleAbs(aligned, alpha=0.65, beta=-10),
        ]
        probe = self.webcam_probe(bgr, row, rng or np.random.default_rng(0))
        return [self.embed_aligned(v) for v in variants], probe

    def webcam_probe(self, bgr: np.ndarray, row: np.ndarray, rng: np.random.Generator) -> Optional[np.ndarray]:
        """
        Re-renders a dataset photo the way a webcam would see that person (small
        face in a 1280x720 frame, dimmer, sensor noise, JPEG) and embeds it through
        the same detect -> align -> embed path used live.
        """
        fx, fy, fw, fh = row[:4]
        s = rng.uniform(65.0, 130.0) / max(1.0, fw)
        small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        canvas = np.full((720, 1280, 3), int(rng.uniform(30, 90)), np.uint8)
        # Place the face near a random point of the frame
        cx, cy = (fx + fw / 2) * s, (fy + fh / 2) * s
        tx, ty = rng.uniform(300, 980), rng.uniform(200, 420)
        ox, oy = int(tx - cx), int(ty - cy)
        sx1, sy1 = max(0, -ox), max(0, -oy)
        dx1, dy1 = max(0, ox), max(0, oy)
        w = min(small.shape[1] - sx1, 1280 - dx1)
        h = min(small.shape[0] - sy1, 720 - dy1)
        if w <= 0 or h <= 0:
            return None
        canvas[dy1:dy1 + h, dx1:dx1 + w] = small[sy1:sy1 + h, sx1:sx1 + w]
        canvas = cv2.convertScaleAbs(canvas, alpha=rng.uniform(0.55, 0.95), beta=rng.uniform(-10, 15))
        canvas = np.clip(canvas + rng.normal(0, rng.uniform(3, 9), canvas.shape), 0, 255).astype(np.uint8)
        q = int(rng.uniform(50, 80))
        canvas = cv2.imdecode(cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)
        faces = self.detect(canvas)
        if len(faces) == 0:
            return None
        near = min(faces, key=lambda f: (f[0] + f[2] / 2 - tx) ** 2 + (f[1] + f[3] / 2 - ty) ** 2)
        if abs(near[0] + near[2] / 2 - tx) > near[2] or abs(near[1] + near[3] / 2 - ty) > near[3]:
            return None
        return self.embed(canvas, near)


def discover_dataset(dataset_dir: str) -> Dict[str, List[str]]:
    """Maps each person (sub-folder name) to their image paths."""
    people: Dict[str, List[str]] = {}
    for sub in sorted(Path(dataset_dir).iterdir()):
        if sub.is_dir():
            files = sorted(str(p) for p in sub.iterdir() if p.suffix.lower() in IMAGE_EXTS)
            if files:
                people[sub.name] = files
    return people


def find_default_dataset_dir() -> Optional[str]:
    base = Path(default_models_dir()).parent / "dataset"
    if not base.is_dir():
        return None
    for candidate in [base, *base.rglob("*")]:
        if candidate.is_dir():
            subdirs = [d for d in candidate.iterdir() if d.is_dir()]
            if len(subdirs) >= 2 and all(any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir()) for d in subdirs):
                return str(candidate)
    return None


def load_image_for_enrollment(path: str) -> Optional[np.ndarray]:
    """Loads a photo at reduced size; JPEG decoding at 1/2 or 1/4 scale is much faster."""
    img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_2)
    if img is not None and max(img.shape[:2]) > 2200:
        img = cv2.imread(path, cv2.IMREAD_REDUCED_COLOR_4)
    elif img is not None and max(img.shape[:2]) < 1000:
        # Small photos (e.g. webcam captures) keep full resolution
        img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        # cv2.imread cannot open non-ASCII Windows paths
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            return None
        if img is not None and max(img.shape[:2]) > 1600:
            s = 1600.0 / max(img.shape[:2])
            img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return img


def build_gallery(
    dataset_dir: str,
    out_path: Optional[str] = None,
    models_dir: Optional[str] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> Dict:
    """
    Builds face_gallery.npz from a dataset folder (one sub-folder per person),
    and calibrates the match percentage shown in the app (see _fit_probability).
    Returns a summary dict.
    """
    ident = FaceIdentifier(models_dir=models_dir, gallery_path="__none__")
    if not ident.models_loaded:
        raise RuntimeError(ident.error or "Face models not available")
    out_path = out_path or os.path.join(ident.models_dir, GALLERY_FILE)

    people = discover_dataset(dataset_dir)
    if not people:
        raise RuntimeError(f"No person folders with images found in {dataset_dir}")
    names = list(people.keys())
    total = sum(len(v) for v in people.values())

    embeddings, labels, photo_idx = [], [], []
    probes, probe_labels, probe_pos = [], [], []
    used: Dict[str, int] = {n: 0 for n in names}
    rng = np.random.default_rng(1234)
    done = 0
    for li, name in enumerate(names):
        for pi, path in enumerate(people[name]):
            img = load_image_for_enrollment(path)
            vecs, probe = ident.enroll_image(img, rng=rng) if img is not None else ([], None)
            p_pos = pi / max(1, len(people[name]) - 1)
            if vecs:
                used[name] += 1
                for v in vecs:
                    embeddings.append(v)
                    labels.append(li)
                    photo_idx.append(p_pos)
            if probe is not None:
                probes.append(probe)
                probe_labels.append(li)
                probe_pos.append(p_pos)
            done += 1
            if progress:
                progress(done, total, name)

    if not embeddings:
        raise RuntimeError("No usable faces were found in the dataset")
    emb = np.stack(embeddings).astype(np.float32)
    lab = np.array(labels, dtype=np.int32)
    pos = np.array(photo_idx, dtype=np.float32)

    threshold, margin = DEFAULT_THRESHOLD, DEFAULT_MARGIN
    stats = _evaluate(emb, lab, pos, names, threshold, margin)
    extra = {}
    if len(probes) >= 20:
        calib, calib_stats = _fit_probability(
            np.stack(probes).astype(np.float32), np.array(probe_labels, np.int32),
            np.array(probe_pos, np.float32), emb, lab, pos, len(names))
        extra["calibration"] = np.array(calib, dtype=np.float32)
        stats["percent_calibration"] = calib_stats
    np.savez_compressed(out_path, embeddings=emb, labels=lab, names=np.array(names),
                        threshold=np.float32(threshold), margin=np.float32(margin), **extra)
    return {
        "gallery_path": out_path,
        "names": names,
        "photos_total": {n: len(people[n]) for n in names},
        "photos_used": used,
        "embeddings": int(emb.shape[0]),
        "threshold": threshold,
        "margin": margin,
        **stats,
    }


def _person_scores(query: np.ndarray, gallery: np.ndarray, gl: np.ndarray, n_people: int, exclude: int = -1) -> np.ndarray:
    """Top-k mean similarity of each query embedding to each person in the gallery."""
    sims = query @ gallery.T
    out = np.full((query.shape[0], n_people), -1.0, dtype=np.float32)
    for p in range(n_people):
        if p == exclude:
            continue
        cols = sims[:, gl == p]
        if cols.shape[1] == 0:
            continue
        k = min(TOP_K, cols.shape[1])
        out[:, p] = np.mean(np.partition(cols, -k, axis=1)[:, -k:], axis=1)
    return out


def _accepts(scores: np.ndarray, threshold: float, margin: float) -> Tuple[np.ndarray, np.ndarray]:
    """Applies the match rule row-wise. Returns (accepted mask, best person index)."""
    srt = np.sort(scores, axis=1)
    best = srt[:, -1]
    second = srt[:, -2] if scores.shape[1] > 1 else np.full(len(scores), -1.0)
    return (best >= threshold) & (best - second >= margin), np.argmax(scores, axis=1)


def _evaluate(emb: np.ndarray, lab: np.ndarray, pos: np.ndarray, names: List[str],
              threshold: float, margin: float) -> Dict:
    """
    Dataset-level sanity check of the match rule:
      - genuine: the last 30% of each person's burst matched against the first 70%
      - stranger: each person matched against a gallery that excludes them
    Note: all photos come from one session, so real webcam scores will be lower.
    """
    n = len(names)
    train = pos <= 0.7
    test = ~train

    scores = _person_scores(emb[test], emb[train], lab[train], n)
    t_lab = lab[test]
    acc, pred = _accepts(scores, threshold, margin)
    holdout_acc = float(np.mean(np.argmax(scores, axis=1) == t_lab)) if len(t_lab) else 0.0

    stranger_accepts = []
    for p in range(n):
        rows = lab == p
        if not np.any(rows):
            continue
        others = lab != p
        s = _person_scores(emb[rows], emb[others], lab[others], n, exclude=p)
        s[:, p] = -1.0
        stranger_accepts.append(_accepts(s, threshold, margin)[0])
    stranger_accepts = np.concatenate(stranger_accepts) if stranger_accepts else np.array([False])

    return {
        "holdout_top1_accuracy": round(holdout_acc, 4),
        "holdout_correct_and_accepted": round(float(np.mean(acc & (pred == t_lab))), 4) if len(t_lab) else 0.0,
        "holdout_wrong_name": round(float(np.mean(acc & (pred != t_lab))), 4) if len(t_lab) else 0.0,
        "stranger_accepted_per_frame": round(float(np.mean(stranger_accepts)), 4),
    }


def _features(scores: np.ndarray) -> np.ndarray:
    srt = np.sort(scores, axis=1)
    second = srt[:, -2] if scores.shape[1] > 1 else np.full(len(scores), -1.0)
    return np.stack([srt[:, -1], srt[:, -1] - second], axis=1)


def _fit_logistic(x: np.ndarray, y: np.ndarray, l2: float = 1e-3, iters: int = 50) -> np.ndarray:
    """Class-balanced logistic regression by Newton's method. Returns [w..., bias]."""
    X = np.hstack([x, np.ones((len(x), 1))])
    sw = np.where(y > 0.5, 0.5 / max(1, y.sum()), 0.5 / max(1, (1 - y).sum()))
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (sw * (p - y)) + l2 * w
        H = (X * (sw * p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def _fit_probability(probes: np.ndarray, plab: np.ndarray, ppos: np.ndarray,
                     emb: np.ndarray, lab: np.ndarray, pos: np.ndarray, n: int) -> Tuple[List[float], Dict]:
    """
    Learns how to turn (similarity, lead over runner-up) into "% chance this is
    the person", using webcam-quality probes of the dataset photos:
      - enrolled case: the later 30% of each burst vs a gallery of the earlier 70%;
        positive only when the best guess is the right person
      - stranger case: every probe vs a gallery that leaves that person out (always negative)
    Both cases get equal weight (a 50/50 prior that a face in view is enrolled).
    """
    train = pos <= 0.7
    held = ppos > 0.7
    s_known = _person_scores(probes[held], emb[train], lab[train], n)
    y_known = (np.argmax(s_known, axis=1) == plab[held]).astype(np.float64)

    s_str = []
    for p in range(n):
        rows = plab == p
        if not np.any(rows):
            continue
        others = lab != p
        sc = _person_scores(probes[rows], emb[others], lab[others], n, exclude=p)
        sc[:, p] = -1.0
        s_str.append(sc)
    s_str = np.concatenate(s_str) if s_str else np.zeros((0, n))

    x = np.vstack([_features(s_known), _features(s_str)])
    y = np.concatenate([y_known, np.zeros(len(s_str))])
    w = _fit_logistic(x, y)
    prob = 1.0 / (1.0 + np.exp(-(np.hstack([x, np.ones((len(x), 1))]) @ w)))

    n_known = len(s_known)
    pk, ps = prob[:n_known], prob[n_known:]
    acc_k = pk >= ACCEPT_PROBABILITY
    # Reliability: of faces shown with ~X%, how many really were that person
    bins = []
    for lo, hi in ((0.5, 0.7), (0.7, 0.9), (0.9, 1.01)):
        m = (prob >= lo) & (prob < hi)
        if np.any(m):
            bins.append({"shown": f"{int(lo * 100)}-{min(100, int(hi * 100))}%", "faces": int(m.sum()),
                         "actually_correct": round(float(np.mean(y[m])), 3)})
    stats = {
        "probes": int(len(probes)),
        "enrolled_named_correctly": round(float(np.mean(acc_k & (y_known > 0.5))), 4) if n_known else 0.0,
        "enrolled_named_wrongly": round(float(np.mean(acc_k & (y_known < 0.5))), 4) if n_known else 0.0,
        "stranger_given_a_name": round(float(np.mean(ps >= ACCEPT_PROBABILITY)), 4) if len(ps) else 0.0,
        "median_percent_correct_person": round(float(np.median(pk[y_known > 0.5])) * 100, 1) if np.any(y_known > 0.5) else 0.0,
        "median_percent_stranger": round(float(np.median(ps)) * 100, 1) if len(ps) else 0.0,
        "reliability": bins,
    }
    return [float(v) for v in w], stats
