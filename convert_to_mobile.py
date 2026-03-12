import sys
from types import ModuleType

import torch
import litert_torch
from face_detector import MobileFaceDetector

model = MobileFaceDetector()
model.load_state_dict(torch.load("mobile_face_detector_epoch_10.pth", map_location="cpu"))
model.eval()

dummy_input = (torch.randn(1, 3, 256, 256),)

torch_out = model(*dummy_input)

print("Converting...")
edge_model = litert_torch.convert(model, dummy_input)
edge_model.export("face_detector.tflite")

# Sanity check
out = edge_model(*dummy_input)
print(f"Output shape: {out.shape}")   # must be (1, 8)
print(f"Output values: {out}")
