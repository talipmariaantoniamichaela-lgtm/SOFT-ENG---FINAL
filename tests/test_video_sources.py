"""
Unit tests for camera enumeration and video source handling.
"""
import pytest
from app.video.camera_manager import CameraManager


def test_camera_enumeration():
    cameras = CameraManager.enumerate_cameras(max_to_test=2)
    assert isinstance(cameras, list)
    assert len(cameras) >= 1
    assert "index" in cameras[0]
    assert "name" in cameras[0]
