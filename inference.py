import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt


class FaceDetector(torch.nn.Module):
    def __init__(self, initial_channels=32):
        super(FaceDetector, self).__init__()

        from torch.nn import Conv2d, BatchNorm2d, ReLU, Sequential, Module

        class DepthwiseSeparableConv(Module):
            def __init__(
                self, in_channels, out_channels, stride=1, expand_channels=None
            ):
                super().__init__()
                self.expand_channels = (
                    expand_channels if expand_channels else in_channels
                )
                self.depthwise = Conv2d(
                    self.expand_channels,
                    self.expand_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    groups=self.expand_channels,
                    bias=False,
                )
                self.bn1 = BatchNorm2d(self.expand_channels)
                self.pointwise = Conv2d(
                    self.expand_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
                self.bn2 = BatchNorm2d(out_channels)

            def forward(self, x):
                x = self.depthwise(x)
                x = self.bn1(x)
                x = ReLU(inplace=True)(x)
                x = self.pointwise(x)
                x = self.bn2(x)
                x = ReLU(inplace=True)(x)
                return x

        self.initial_rgb = Sequential(
            Conv2d(3, initial_channels, kernel_size=3, stride=2, padding=1, bias=False),
            BatchNorm2d(initial_channels),
            ReLU(inplace=True),
        )

        self.block1 = DepthwiseSeparableConv(initial_channels, 64, 1, initial_channels)
        self.block2 = DepthwiseSeparableConv(64, 128, 2, 64)
        self.block3 = DepthwiseSeparableConv(128, 128, 1, None)
        self.block4 = DepthwiseSeparableConv(128, 256, 2, 128)
        self.block5 = DepthwiseSeparableConv(256, 256, 1, None)

        self.detection_head = Sequential(
            Conv2d(256, 9, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(9),
        )

    def forward(self, x):
        x = self.initial_rgb(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.detection_head(x)
        return x


def run_inference(model_path, image_path, device_type="cuda"):
    device = torch.device(device_type if torch.cuda.is_available() else "cpu")

    model = FaceDetector()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    orig_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_img.size

    img_tensor = transform(orig_img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        outputs_flat = (
            outputs.view(outputs.size(0), 9, -1).mean(dim=2).squeeze(0).cpu().numpy()
        )

    x_norm, y_norm, w_norm, h_norm, conf, lex_norm, ley_norm, rex_norm, rey_norm = (
        outputs_flat
    )

    x = max(0, int(x_norm * orig_w))
    y = max(0, int(y_norm * orig_h))
    w = max(0, int(w_norm * orig_w))
    h = max(0, int(h_norm * orig_h))

    left_eye = (int(lex_norm * orig_w), int(ley_norm * orig_h))
    right_eye = (int(rex_norm * orig_w), int(rey_norm * orig_h))

    img_np = np.array(orig_img)

    cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.circle(img_np, left_eye, 5, (255, 0, 0), -1)
    cv2.circle(img_np, right_eye, 5, (0, 0, 255), -1)

    return img_np, {
        "bbox": (x, y, w, h),
        "left_eye": left_eye,
        "right_eye": right_eye,
        "confidence": conf,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = "face_detector.pth"

    result_img, info = run_inference(model_path, image_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(result_img)
    plt.axis("off")
    plt.title(f"Confidence: {info['confidence']:.4f}")
    plt.show()
