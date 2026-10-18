"""
Camera Manager and Video Device Enumerator for Motion Reader.
"""
import cv2
from typing import List, Dict, Any, Optional


class CameraManager:
    """
    Manages camera device discovery, initialization, and resource cleanup.
    """
    @staticmethod
    def enumerate_cameras(max_to_test: int = 4) -> List[Dict[str, Any]]:
        """
        Scans indices to find available connected webcams.
        Returns list of dicts: [{"index": 0, "name": "Camera (Default)"}, ...]
        """
        cameras = []
        for idx in range(max_to_test):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        name = "Camera (Default)" if idx == 0 else f"Camera {idx}"
                        cameras.append({"index": idx, "name": name})
                    cap.release()
            except Exception:
                pass

        if not cameras:
            # Fallback if detection is slow/restricted
            cameras.append({"index": 0, "name": "Camera (Default)"})

        return cameras

    @staticmethod
    def create_capture(camera_index: int, width: int = 1280, height: int = 720) -> Optional[cv2.VideoCapture]:
        """
        Opens camera capture with requested resolution.
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                return cap
        except Exception as e:
            print(f"Failed to open camera {camera_index}: {e}")
        return None
