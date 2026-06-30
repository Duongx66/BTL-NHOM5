Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$pythonExe = "C:\Users\Duong\AppData\Local\Python\pythoncore-3.14-64\python.exe"
$datasetZip = "D:\Dowload " + [char]0x1ED5 + " D\archive.zip"

& $pythonExe -m src.prepare_data --zip-path $datasetZip --output-dir data

& $pythonExe -m src.train --model baseline_cnn --epochs 15 --batch-size 32 --run-name run1_baseline --wandb-mode online
& $pythonExe -m src.train --model resnet18 --pretrained --epochs 15 --batch-size 32 --run-name run2_resnet18 --wandb-mode online
& $pythonExe -m src.train --model mobilenet_v2 --pretrained --epochs 15 --batch-size 32 --run-name run3_mobilenet --wandb-mode online
& $pythonExe -m src.train --model efficientnet_b0 --pretrained --epochs 15 --batch-size 32 --run-name run4_efficientnet --wandb-mode online
& $pythonExe -m src.train --model mobilenet_v2 --pretrained --weighted-sampler --epochs 15 --batch-size 32 --run-name run5_mobilenet_weighted --wandb-mode online

& $pythonExe -m src.evaluate --checkpoint outputs/checkpoints/best.pt
& $pythonExe -m src.export_onnx --checkpoint outputs/checkpoints/best.pt --onnx-path outputs/models/best.onnx
