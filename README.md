
# CSC4005 Final Project - Topic 8

Topic 8: Image-based waste classification for a greener environment.

This project uses the Kaggle Garbage Classification dataset with 6 classes:
`cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

## Project Contents

- Baseline CNN built from scratch.
- Transfer learning with ResNet18, MobileNetV2, and EfficientNet-B0.
- Data augmentation, stratified train/validation/test split.
- Class imbalance handling with weighted sampler or class weights.
- W&B logging for at least 5 runs.
- Evaluation with accuracy, macro-F1, precision, recall, confusion matrix, and wrong prediction analysis.
- Grad-CAM for model explainability.
- ONNX export and ONNX Runtime inference test.
- Streamlit demo for image upload and top-3 prediction.

## Dataset

Default zip path:

```powershell
D:\Dowload ổ D\archive.zip
```

Prepare the dataset:

```powershell
python -m src.prepare_data --zip-path "D:\Dowload ổ D\archive.zip" --output-dir data
```

The script creates:

```text
data/
  raw/
  splits/
    train.csv
    val.csv
    test.csv
    classes.json
```

## Train

Login to W&B first. On this machine, `wandb` may not be in PATH, so use:

```powershell
.\scripts\wandb_login.ps1
```

Then paste your API key from https://wandb.ai/authorize.

Baseline CNN:

```powershell
python -m src.train --model baseline_cnn --epochs 15 --batch-size 32 --run-name baseline_cnn
```

Transfer learning examples:

```powershell
python -m src.train --model resnet18 --pretrained --epochs 15 --batch-size 32 --run-name resnet18_pretrained
python -m src.train --model mobilenet_v2 --pretrained --epochs 15 --batch-size 32 --run-name mobilenet_v2_pretrained
python -m src.train --model efficientnet_b0 --pretrained --epochs 15 --batch-size 32 --run-name efficientnet_b0_pretrained
```

Suggested 5 W&B runs:

```powershell
python -m src.train --model baseline_cnn --epochs 15 --run-name run1_baseline
python -m src.train --model resnet18 --pretrained --epochs 15 --run-name run2_resnet18
python -m src.train --model mobilenet_v2 --pretrained --epochs 15 --run-name run3_mobilenet
python -m src.train --model efficientnet_b0 --pretrained --epochs 15 --run-name run4_efficientnet
python -m src.train --model mobilenet_v2 --pretrained --weighted-sampler --epochs 15 --run-name run5_mobilenet_weighted
```

Or run all 5 experiments with the prepared PowerShell script:

```powershell
./scripts/run_5_experiments.ps1
```

Use `--wandb-mode offline` if you do not want to sync immediately.

## Evaluate

```powershell
python -m src.evaluate --checkpoint outputs/checkpoints/best.pt
```

Outputs are saved to `outputs/evaluation/`, including:

- `metrics.json`
- `confusion_matrix.png`
- `wrong_predictions.csv`
- Grad-CAM images in `outputs/evaluation/gradcam/`

## Export ONNX

```powershell
python -m src.export_onnx --checkpoint outputs/checkpoints/best.pt --onnx-path outputs/models/best.onnx
python -m src.infer_onnx --onnx-path outputs/models/best.onnx --image-path path\to\image.jpg
```

## Demo

```powershell
python -m streamlit run app.py
```

If the `streamlit` command is not found, use the Python module form above.

Then upload a waste image and view top-3 predictions.
