import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Generate Grad-CAM visualizations for the garbage classifier.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/evaluation/gradcam")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradcam-samples", type=int, default=8)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.evaluate",
        "--checkpoint",
        args.checkpoint,
        "--output-dir",
        args.output_dir,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--gradcam-samples",
        str(args.gradcam_samples),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
