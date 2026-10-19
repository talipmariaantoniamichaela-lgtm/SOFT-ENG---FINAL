"""Small, explainable facial-expression classifier for MediaPipe blendshapes."""
from typing import Dict, Tuple


def classify_expression(scores: Dict[str, float]) -> Tuple[str, float]:
    """Return a conservative expression label and confidence in the 0–1 range.

    MediaPipe produces facial *blendshapes*, not emotion labels.  These rules
    deliberately report only broad, observable expressions and fall back to
    Neutral when no signal is strong enough.
    """
    def avg(*names: str) -> float:
        return sum(float(scores.get(name, 0.0)) for name in names) / len(names)

    smile = avg("mouthSmileLeft", "mouthSmileRight")
    frown = avg("mouthFrownLeft", "mouthFrownRight")
    brow_down = avg("browDownLeft", "browDownRight")
    eye_wide = avg("eyeWideLeft", "eyeWideRight")
    brow_up = max(float(scores.get("browInnerUp", 0.0)), avg("browOuterUpLeft", "browOuterUpRight"))
    jaw_open = float(scores.get("jawOpen", 0.0))
    pucker = float(scores.get("mouthPucker", 0.0))

    # Evaluate distinctive combinations first so an open, surprised mouth is
    # not misclassified as a neutral expression.
    surprise = (brow_up + eye_wide + jaw_open) / 3.0
    if surprise >= 0.38 and jaw_open >= 0.20:
        return "Surprised", min(1.0, surprise)
    if smile >= 0.30:
        return "Smiling", min(1.0, smile)
    if frown >= 0.28 and brow_up >= 0.12:
        return "Sad", min(1.0, (frown + brow_up) / 2.0)
    if brow_down >= 0.32 and frown < 0.25:
        return "Frowning", min(1.0, brow_down)
    if pucker >= 0.38:
        return "Puckering", min(1.0, pucker)
    return "Neutral", max(0.0, 1.0 - max(smile, frown, brow_down, surprise, pucker))
