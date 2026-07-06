# tests/test_model.py
import pytest
from ultralytics import YOLO

def test_model_loads():
    model = YOLO("yolov8n-seg.pt")
    assert model is not None
