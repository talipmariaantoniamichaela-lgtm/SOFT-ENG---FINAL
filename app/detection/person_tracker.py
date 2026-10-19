"""
Multi-person tracking for Motion Reader.

Keeps a stable track ID for every person across frames, owns one landmark
smoother and one MotionAnalyzer per person, and attaches face-recognition
results to the right person via the pose's head landmarks. Identity is decided
by voting over recent recognitions so a single bad frame can't flip a name.
"""
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from app.detection.face_identity import FaceMatch
from app.detection.motion_analyzer import FrameAnalysisResult, MotionAnalyzer
from app.detection.pose_estimator import (
    LandmarkSmoother, TrackingResult, compute_bounding_box,
    NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR,
)

BBox = Tuple[float, float, float, float]

ID_VOTE_WINDOW = 15      # recent recognitions considered per person
ID_MIN_VOTES = 3         # agreeing recognitions needed before a name is shown


@dataclass
class PersonTrack:
    track_id: int
    smoother: LandmarkSmoother
    analyzer: MotionAnalyzer
    bbox: BBox
    misses: int = 0
    age: int = 0
    # Recent recognitions: (accepted name or None, similarity, best guess, match probability)
    id_votes: Deque[Tuple[Optional[str], float, str, float]] = field(default_factory=lambda: deque(maxlen=ID_VOTE_WINDOW))
    name: Optional[str] = None
    name_score: float = 0.0
    match_percent: float = 0.0                   # % chance this person is `name` (or the closest guess)
    closest_name: Optional[str] = None           # best guess while unrecognized
    face_box: Optional[BBox] = None              # normalized (x1, y1, x2, y2), last seen face
    tracking: Optional[TrackingResult] = None
    analysis: Optional[FrameAnalysisResult] = None

    def vote_summary(self) -> Tuple[Optional[str], float, float]:
        """Returns (winning name or None, mean similarity of its votes, its vote weight)."""
        totals: Dict[str, List[float]] = {}
        for name, score, _, _ in self.id_votes:
            if name:
                totals.setdefault(name, []).append(score)
        if not totals:
            return None, 0.0, 0.0
        best = max(totals, key=lambda n: sum(totals[n]))
        votes = totals[best]
        if len(votes) < ID_MIN_VOTES or len(votes) < 0.4 * len(self.id_votes):
            return None, 0.0, 0.0
        return best, sum(votes) / len(votes), sum(votes)

    def percent_for(self, name: str) -> float:
        """
        Match percentage for `name` averaged over recent recognitions; a recognition
        whose best guess was someone else counts as 0%, so inconsistent faces score low.
        """
        if not self.id_votes:
            return 0.0
        return 100.0 * sum(p if g == name else 0.0 for _, _, g, p in self.id_votes) / len(self.id_votes)

    def closest_guess(self) -> Optional[str]:
        weights: Dict[str, float] = {}
        for _, _, guess, prob in self.id_votes:
            if guess:
                weights[guess] = weights.get(guess, 0.0) + prob
        return max(weights, key=weights.get) if weights else None


@dataclass
class PersonSnapshot:
    """Immutable per-frame view of one person, safe to hand to the UI thread."""
    track_id: int
    name: Optional[str]
    name_score: float
    tracking: TrackingResult
    analysis: FrameAnalysisResult
    face_box: Optional[BBox]
    match_percent: float = 0.0           # for `name`, or for `closest_name` when unrecognized
    closest_name: Optional[str] = None
    face_seen: bool = False              # at least one recognition attempt on this person
    facial_expression: str = "Neutral"
    facial_expression_confidence: float = 0.0

    @property
    def label(self) -> str:
        return self.name or f"Person {self.track_id}"


def _iou(a: BBox, b: BBox) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _center(b: BBox) -> Tuple[float, float]:
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


class PersonTracker:
    def __init__(
        self,
        sensitivity: float = 1.0,
        smoothing: bool = True,
        smoothing_factor: float = 0.55,
        max_misses: int = 12,
    ):
        self.sensitivity = sensitivity
        self.smoothing = smoothing
        self.smoothing_factor = smoothing_factor
        self.max_misses = max_misses
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.next_id = 1

    def set_sensitivity(self, sensitivity: float) -> None:
        self.sensitivity = sensitivity
        for tr in self.tracks.values():
            tr.analyzer.sensitivity = max(0.5, min(2.0, sensitivity))

    # ------------------------------------------------------------------
    def update(self, detections: List[TrackingResult], t: float) -> Tuple[List[PersonTrack], List[PersonTrack]]:
        """
        Matches this frame's pose detections to existing tracks, runs per-person
        smoothing and motion analysis. Returns (visible tracks, tracks that just left).
        """
        pairs = []
        for tid, track in self.tracks.items():
            for di, det in enumerate(detections):
                iou = _iou(track.bbox, det.bounding_box)
                tc, dc = _center(track.bbox), _center(det.bounding_box)
                size = max(track.bbox[2] - track.bbox[0], track.bbox[3] - track.bbox[1], 0.05)
                d = math.hypot(tc[0] - dc[0], tc[1] - dc[1]) / size
                if iou > 0.15 or d < 0.6:
                    pairs.append((iou - 0.25 * d, tid, di))
        pairs.sort(reverse=True)

        matched_tracks, matched_dets = set(), set()
        assignment: Dict[int, int] = {}
        for _, tid, di in pairs:
            if tid in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(tid)
            matched_dets.add(di)
            assignment[tid] = di

        for di, det in enumerate(detections):
            if di not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = PersonTrack(
                    track_id=tid,
                    smoother=LandmarkSmoother(alpha=self.smoothing_factor),
                    analyzer=MotionAnalyzer(sensitivity=self.sensitivity),
                    bbox=det.bounding_box,
                )
                assignment[tid] = di

        visible, departed = [], []
        for tid in list(self.tracks):
            track = self.tracks[tid]
            if tid in assignment:
                det = detections[assignment[tid]]
                if self.smoothing:
                    det.pose_landmarks = track.smoother.smooth(det.pose_landmarks)
                    det.bounding_box = compute_bounding_box(det.pose_landmarks) or det.bounding_box
                det.person_id = tid
                track.bbox = det.bounding_box
                track.misses = 0
                track.age += 1
                track.tracking = det
                track.analysis = track.analyzer.analyze(det, t)
                track.analysis.person_id = tid
                visible.append(track)
            else:
                track.misses += 1
                track.tracking = None
                track.analysis = None
                if track.misses > self.max_misses:
                    departed.append(self.tracks.pop(tid))
        return visible, departed

    # ------------------------------------------------------------------
    def assign_faces(self, faces: List[FaceMatch], frame_w: int, frame_h: int, visible: List[PersonTrack]) -> None:
        """Attaches each recognized face to the visible person whose head is inside the face box."""
        claimed = set()
        for face in sorted(faces, key=lambda f: f.box[2] * f.box[3], reverse=True):
            x, y, w, h = face.box
            pad_x, pad_y = 0.35 * w, 0.35 * h
            fx1, fy1 = (x - pad_x) / frame_w, (y - pad_y) / frame_h
            fx2, fy2 = (x + w + pad_x) / frame_w, (y + h + pad_y) / frame_h
            fcx, fcy = (x + w / 2) / frame_w, (y + h / 2) / frame_h

            best, best_d = None, float("inf")
            for track in visible:
                if track.track_id in claimed or track.tracking is None:
                    continue
                lms = track.tracking.pose_landmarks
                head = [lms[i] for i in (NOSE, LEFT_EYE, RIGHT_EYE, LEFT_EAR, RIGHT_EAR) if lms[i].visibility > 0.3]
                if not head:
                    continue
                inside = [p for p in head if fx1 <= p.x <= fx2 and fy1 <= p.y <= fy2]
                if not inside:
                    continue
                hx = sum(p.x for p in inside) / len(inside)
                hy = sum(p.y for p in inside) / len(inside)
                d = math.hypot(hx - fcx, hy - fcy)
                if d < best_d:
                    best, best_d = track, d
            if best is None:
                continue
            claimed.add(best.track_id)
            best.id_votes.append((face.name, face.score, face.best_guess, face.probability))
            best.face_box = (x / frame_w, y / frame_h, (x + w) / frame_w, (y + h) / frame_h)

        self._resolve_names()

    def _resolve_names(self) -> None:
        """Updates every track's name; one name can belong to only one person at a time."""
        proposals = []
        for track in self.tracks.values():
            name, score, weight = track.vote_summary()
            proposals.append((weight, track, name, score))
        taken = set()
        for weight, track, name, score in sorted(proposals, key=lambda p: p[0], reverse=True):
            if name and name not in taken:
                taken.add(name)
                track.name, track.name_score = name, round(score, 2)
                track.closest_name = None
                track.match_percent = round(track.percent_for(name), 1)
            else:
                track.name, track.name_score = None, 0.0
                guess = track.closest_guess()
                # A name already shown on someone else can't also be this person's guess
                track.closest_name = guess if guess not in taken else None
                track.match_percent = round(track.percent_for(guess), 1) if track.closest_name else 0.0

    # ------------------------------------------------------------------
    @staticmethod
    def snapshot(visible: List[PersonTrack]) -> List[PersonSnapshot]:
        out = []
        for track in visible:
            if track.tracking is None or track.analysis is None:
                continue
            track.analysis.subject_name = track.name
            track.analysis.subject_confidence = round(track.match_percent / 100.0, 2) if track.name else None
            out.append(PersonSnapshot(
                track_id=track.track_id,
                name=track.name,
                name_score=track.name_score,
                tracking=track.tracking,
                analysis=track.analysis,
                face_box=track.face_box,
                match_percent=track.match_percent,
                closest_name=track.closest_name,
                face_seen=bool(track.id_votes),
                facial_expression=track.tracking.facial_expression,
                facial_expression_confidence=track.tracking.facial_expression_confidence,
            ))
        return out
