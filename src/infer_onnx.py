import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from .data import get_transforms


def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    exp = np.exp(x)
    return exp / exp.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--classes-path", default="data/splits/classes.json")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    classes = json.loads(Path(args.classes_path).read_text(encoding="utf-8"))
    transform = get_transforms(args.image_size, train=False)
    image = Image.open(args.image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).numpy().astype(np.float32)

    session = ort.InferenceSession(args.onnx_path, providers=["CPUExecutionProvider"])
    logits = session.run(["logits"], {"image": tensor})[0]
    probs = softmax(logits)[0]
    top3 = probs.argsort()[-3:][::-1]
    for idx in top3:
        print(f"{classes[idx]}: {probs[idx]:.4f}")


if __name__ == "__main__":
    main()
