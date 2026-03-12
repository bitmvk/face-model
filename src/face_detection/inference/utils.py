import torch
import torchvision.transforms as transforms
from PIL import Image

TARGET_SIZE = 256


def letterbox_image(image, target_size=None):
    """
    Resize image with letterboxing to target_size and return tensor + parameters.
    """
    if target_size is None:
        target_size = TARGET_SIZE

    orig_w, orig_h = image.size
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    resized = image.resize((new_w, new_h), Image.BILINEAR)

    padded = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    pad_left = (target_size - new_w) // 2
    pad_top = (target_size - new_h) // 2
    padded.paste(resized, (pad_left, pad_top))

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    tensor = transform(padded)
    return tensor, scale, pad_left, pad_top


def extract_detection(
    outputs, orig_w, orig_h, scale, pad_left, pad_top, target_size=None
):
    """
    Convert model outputs (normalized in padded image) to original image coordinates.
    """
    if target_size is None:
        target_size = TARGET_SIZE

    outputs = outputs.squeeze(0).cpu().numpy()
    x_norm, y_norm, w_norm, h_norm, lex_norm, ley_norm, rex_norm, rey_norm = outputs

    x_pad = x_norm * target_size
    y_pad = y_norm * target_size
    w_pad = w_norm * target_size
    h_pad = h_norm * target_size
    lex_pad = lex_norm * target_size
    ley_pad = ley_norm * target_size
    rex_pad = rex_norm * target_size
    rey_pad = rey_norm * target_size

    x_orig = int((x_pad - pad_left) / scale)
    y_orig = int((y_pad - pad_top) / scale)
    w_orig = int(w_pad / scale)
    h_orig = int(h_pad / scale)
    lex_orig = int((lex_pad - pad_left) / scale)
    ley_orig = int((ley_pad - pad_top) / scale)
    rex_orig = int((rex_pad - pad_left) / scale)
    rey_orig = int((rey_pad - pad_top) / scale)

    conf = 1.0
    return (
        x_orig,
        y_orig,
        w_orig,
        h_orig,
        conf,
        (lex_orig, ley_orig),
        (rex_orig, rey_orig),
    )
