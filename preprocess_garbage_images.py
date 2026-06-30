import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Prepare the garbage dataset.")
    parser.add_argument("--zip-path", default=r"D:\Dowload ổ D\archive.zip")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--val-size", type=float, default=0.15)
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "-m",
        "src.prepare_data",
        "--zip-path",
        args.zip_path,
        "--output-dir",
        args.output_dir,
        "--seed",
        str(args.seed),
        "--test-size",
        str(args.test_size),
        "--val-size",
        str(args.val_size),
    ]
    subprocess.check_call(cmd)


if __name__ == "__main__":
    main()
