import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

from .data import WasteDataset, compute_class_weights, get_transforms, load_classes, make_weighted_sampler
from .metrics import compute_metrics
from .models import build_model


def run_epoch(model, loader, criterion, optimizer, device, train=False):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []
    for images, labels, _ in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
    return total_loss / len(loader.dataset), y_true, y_pred


def save_checkpoint(path, model, args, classes, metrics):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_name": args.model,
            "image_size": args.image_size,
            "classes": classes,
            "metrics": metrics,
            "pretrained": args.pretrained,
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir", default="data/splits")
    parser.add_argument("--model", default="baseline_cnn", choices=["baseline_cnn", "resnet18", "mobilenet_v2", "efficientnet_b0"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--weighted-sampler", action="store_true")
    parser.add_argument("--class-weights", action="store_true")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default="csc4005-topic8-garbage-classification")
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes, class_to_idx = load_classes(args.splits_dir)
    num_classes = len(classes)

    train_dataset = WasteDataset(Path(args.splits_dir) / "train.csv", class_to_idx, get_transforms(args.image_size, train=True))
    val_dataset = WasteDataset(Path(args.splits_dir) / "val.csv", class_to_idx, get_transforms(args.image_size, train=False))
    sampler = make_weighted_sampler(train_dataset) if args.weighted_sampler else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(args.model, num_classes, pretrained=args.pretrained).to(device)
    class_weights = compute_class_weights(Path(args.splits_dir) / "train.csv", class_to_idx).to(device) if args.class_weights else None
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        mode=args.wandb_mode,
        config=vars(args) | {"classes": classes, "device": str(device)},
    )

    best_macro_f1 = -1.0
    best_path = Path(args.output_dir) / "checkpoints" / "best.pt"
    last_path = Path(args.output_dir) / "checkpoints" / "last.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_true, train_pred = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_true, val_pred = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        train_metrics = compute_metrics(train_true, train_pred, classes)
        val_metrics = compute_metrics(val_true, val_pred, classes)
        log_data = {
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "train/accuracy": train_metrics["accuracy"],
            "train/macro_f1": train_metrics["macro_f1"],
            "val/accuracy": val_metrics["accuracy"],
            "val/macro_f1": val_metrics["macro_f1"],
            "lr": scheduler.get_last_lr()[0],
        }
        wandb.log(log_data)
        print(json.dumps(log_data, indent=2))

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            save_checkpoint(best_path, model, args, classes, val_metrics)

    save_checkpoint(last_path, model, args, classes, {"best_val_macro_f1": best_macro_f1})
    if run:
        wandb.finish()
    print(f"Best checkpoint: {best_path}")


if __name__ == "__main__":
    main()
