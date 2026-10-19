from app.detection.facial_expression import classify_expression


def test_smile_is_detected_from_mouth_blendshapes():
    label, confidence = classify_expression({"mouthSmileLeft": 0.8, "mouthSmileRight": 0.7})
    assert label == "Smiling"
    assert confidence > 0.7


def test_surprise_requires_an_open_jaw_and_wide_eyes():
    label, confidence = classify_expression({
        "browInnerUp": 0.7, "eyeWideLeft": 0.7, "eyeWideRight": 0.6, "jawOpen": 0.8,
    })
    assert label == "Surprised"
    assert confidence > 0.6


def test_low_scores_are_neutral():
    assert classify_expression({"mouthSmileLeft": 0.1, "mouthSmileRight": 0.1})[0] == "Neutral"
