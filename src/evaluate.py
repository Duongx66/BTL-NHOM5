import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from .data import WasteDataset, get_transforms, load_classes
from .gradcam import GradCAM, overlay_cam
from .metrics import compute_metrics, save_confusion_matrix, save_metrics, top_confusions
from .models import build_model, get_gradcam_target_layer


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    image = (tensor.cpu() * std + mean).clamp(0, 1)
    return (image.permute(1, 2, 0).numpy() * 255).astype("uint8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--gradcam-samples", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["classes"]
    _, class_to_idx = load_classes(args.splits_dir)
    idx_to_class = {idx: label for label, idx in class_to_idx.items()}

    model = build_model(ckpt["model_name"], len(classes), pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dataset = WasteDataset(Path(args.splits_dir) / "test.csv", class_to_idx, get_transforms(ckpt.get("image_size", 224), train=False))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    y_true, y_pred, paths, probs_all = [], [], [], []
    with torch.no_grad():
        for images, labels, batch_paths in loader:
            images = images.to(device)
            logits = model(images)
            probs = logits.softmax(dim=1).cpu()
            y_true.extend(labels.tolist())
            y_pred.extend(probs.argmax(dim=1).tolist())
            probs_all.extend(probs.tolist())
            paths.extend(batch_paths)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = compute_metrics(y_true, y_pred, classes)
    save_metrics(metrics, output_dir / "metrics.json")
    cm = save_confusion_matrix(y_true, y_pred, classes, output_dir / "confusion_matrix.png")

    rows = []
    for true_idx, pred_idx, path, probs in zip(y_true, y_pred, paths, probs_all):
        if true_idx != pred_idx:
            top3 = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:3]
            rows.append(
                {
                    "path": path,
                    "true": idx_to_class[true_idx],
                    "pred": idx_to_class[pred_idx],
                    "pred_prob": probs[pred_idx],
                    "top3": "; ".join(f"{idx_to_class[i]}={p:.3f}" for i, p in top3),
                }
            )
    pd.DataFrame(rows).to_csv(output_dir / "wrong_predictions.csv", index=False)

    summary = {
        "metrics": metrics,
        "top_confusions": top_confusions(cm, classes, top_k=8),
    }
    (output_dir / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    cam_dir = output_dir / "gradcam"
    cam_dir.mkdir(parents=True, exist_ok=True)
    target_layer = get_gradcam_target_layer(model, ckpt["model_name"])
    gradcam = GradCAM(model, target_layer)
    sample_count = min(args.gradcam_samples, len(dataset))
    for i in range(sample_count):
        image_tensor, label, image_path = dataset[i]
        cam, pred_idx = gradcam(image_tensor.unsqueeze(0).to(device))
        rgb = denormalize(image_tensor)
        overlay = overlay_cam(rgb, cam)
        plt.imsave(cam_dir / f"{i:02d}_{idx_to_class[label]}_pred_{idx_to_class[pred_idx]}.png", overlay)
    gradcam.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
