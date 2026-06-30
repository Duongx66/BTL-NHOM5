import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Train a transfer learning model.")
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v2", "efficientnet_b0"], default="mobilenet_v2")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-mode", default="offline", choices=["online", "offline", "disabled"])
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    run_name = args.run_name or f"{args.model}_pretrained"
    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--model",
        args.model,
        "--pretrained",
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--image-size",
        str(args.image_size),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--run-name",
        run_name,
        "--wandb-mode",
        args.wandb_mode,
        "--num-workers",
        str(args.num_workers),
        "--output-dir",
        args.output_dir,
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
