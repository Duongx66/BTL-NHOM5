# Báo cáo cuối kỳ CSC4005 - Topic 8

## 1. Tổng quan

Đề tài: Phân loại rác thải bằng ảnh.
Mục tiêu: xây dựng hệ thống phân loại ảnh rác thải thành 6 lớp: `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`.

## 2. Dữ liệu và phân chia

Dataset hiện đã chuẩn bị tại `data/` và được chia stratified thành:
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `data/splits/classes.json`

Tỷ lệ train/val/test theo mặc định 70/15/15 với seed 42.

## 3. Mô hình tốt nhất hiện tại

Checkpoint tốt nhất: `outputs/checkpoints/best.pt`
Mô hình sử dụng: theo checkpoint đã lưu trong file.

## 4. Kết quả đánh giá trên test set

File đánh giá: `outputs/evaluation/metrics.json`

- Accuracy: **0.8868**
- Macro F1: **0.8670**
- Macro Precision: **0.8877**
- Macro Recall: **0.8542**

Per-class:
- `cardboard`: precision 0.9825, recall 0.9180, f1 0.9492
- `glass`: precision 0.8904, recall 0.8667, f1 0.8784
- `metal`: precision 0.8769, recall 0.9194, f1 0.8976
- `paper`: precision 0.8854, recall 0.9551, f1 0.9189
- `plastic`: precision 0.8243, recall 0.8472, f1 0.8356
- `trash`: precision 0.8667, recall 0.6190, f1 0.7222

## 5. Phân tích lỗi chính

File phân tích: `outputs/evaluation/analysis_summary.json`

Các cặp nhầm lẫn nhiều nhất:
1. `glass` → `plastic`: 7 lần
2. `plastic` → `glass`: 6 lần
3. `cardboard` → `paper`: 4 lần
4. `trash` → `paper`: 4 lần

Gợi ý nguyên nhân:
- `glass` và `plastic` dễ nhầm khi vật thể trong suốt/ánh sáng giống nhau.
- `cardboard` và `paper` có texture, màu sắc tương tự.
- `trash` ít ảnh và đa dạng nên recall thấp nhất.

## 6. Grad-CAM và giải thích

Ảnh Grad-CAM đã lưu tại `outputs/evaluation/gradcam/`.

Bạn có thể xem các ảnh này để xác thực vùng mô hình đã tập trung khi phân loại các mẫu sai và đúng.

## 7. ONNX

Mô hình đã được export thành công sang:
- `outputs/models/best.onnx`

Kiểm tra ONNX inference bằng ảnh mẫu đã thực hiện với `--image-size 128`:
- Kết quả top-3 dự đoán: `plastic`, `glass`, `paper`

## 8. Chạy 5 thí nghiệm W&B

Script hiện có:
- `scripts/run_5_experiments.ps1`

Nội dung script thực hiện:
- Chuẩn bị dữ liệu từ file zip
- Huấn luyện 5 mô hình:
  1. `baseline_cnn`
  2. `resnet18` pretrained
  3. `mobilenet_v2` pretrained
  4. `efficientnet_b0` pretrained
  5. `mobilenet_v2` pretrained + weighted sampler
- Đánh giá checkpoint tốt nhất
- Export ONNX

Chạy script bằng PowerShell:

```powershell
cd "c:\Users\Duong\OneDrive - Dai Nam University\Desktop\duong\garbage-classification"
./scripts/run_5_experiments.ps1
```

Nếu cần chạy offline, bạn có thể chỉnh `--wandb-mode` trong script thành `offline` hoặc `disabled`.

## 9. Kết luận và hướng tiếp theo

- Mô hình hiện tại đạt accuracy ~88.7% và macro F1 ~86.7%.
- Lớp `trash` là điểm yếu nhất do số lượng mẫu nhỏ và tính đa dạng cao.
- Nên thử thêm augmentation đặc hiệu, balanced sampling hoặc class weights để cải thiện recall cho lớp `trash`.
- Tiếp theo có thể thực hiện:
  - chọn mô hình tốt nhất và chạy test lại với 5 experiment đầy đủ
  - hoàn thiện phần báo cáo / demo Streamlit
  - thêm validation trực quan bằng Grad-CAM và phân tích sai lệch