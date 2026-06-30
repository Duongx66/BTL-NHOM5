# Báo cáo cuối kỳ CSC4005 - Đề tài 8

## 1. Thông tin đề tài

Tên đề tài: Phân loại rác thải bằng ảnh phục vụ môi trường xanh.

Mục tiêu: xây dựng hệ thống phân loại ảnh rác thải thành 6 lớp: cardboard, glass, metal, paper, plastic, trash.

## 2. Dữ liệu

Dataset: Garbage Classification Dataset từ Kaggle.

Phân bố lớp trong file zip đã kiểm tra:

| Lớp | Số ảnh |
|---|---:|
| cardboard | 806 |
| glass | 1002 |
| metal | 820 |
| paper | 1188 |
| plastic | 964 |
| trash | 274 |

Nhận xét: lớp `trash` ít hơn đáng kể, vì vậy cần thử class weights hoặc weighted sampler.

## 3. Pipeline

1. Giải nén dataset và tạo stratified split train/validation/test.
2. Tiền xử lý ảnh về kích thước 224x224.
3. Augmentation: random resized crop, horizontal flip, rotation, color jitter.
4. Huấn luyện baseline CNN tự xây.
5. Huấn luyện transfer learning với ít nhất 2 backbone.
6. Đánh giá bằng accuracy, macro-F1, precision, recall, confusion matrix.
7. Phân tích ảnh dự đoán sai và các cặp lớp dễ nhầm.
8. Export mô hình tốt nhất sang ONNX.
9. Xây dựng demo Streamlit upload ảnh và trả về top-3 prediction.

## 4. Thí nghiệm W&B

| Run | Model | Pretrained | Imbalance Handling | Accuracy | Macro-F1 |
|---|---|---:|---|---:|---:|
| run1_baseline | baseline_cnn | No | None |  |  |
| run2_resnet18 | resnet18 | Yes | None |  |  |
| run3_mobilenet | mobilenet_v2 | Yes | None |  |  |
| run4_efficientnet | efficientnet_b0 | Yes | None |  |  |
| run5_mobilenet_weighted | mobilenet_v2 | Yes | Weighted sampler |  |  |

## 5. Kết quả và phân tích lỗi

Điền kết quả từ `outputs/evaluation/metrics.json`, `confusion_matrix.png`, và `wrong_predictions.csv`.

Các cặp lớp cần chú ý:

- plastic và glass: có thể nhầm khi vật thể trong suốt hoặc nền sáng.
- paper và cardboard: texture và màu sắc gần nhau.
- metal và plastic: một số lon/chai có bề mặt phản chiếu hoặc hình dạng giống nhau.

## 6. Grad-CAM

Chèn ảnh từ `outputs/evaluation/gradcam/` và nhận xét vùng ảnh mô hình tập trung.

## 7. ONNX và demo

Mô hình tốt nhất được export sang `outputs/models/best.onnx`.

Demo Streamlit chạy bằng:

```powershell
streamlit run app.py
```

## 8. Kết luận

Tóm tắt mô hình tốt nhất, ưu nhược điểm, các lớp khó, và hướng cải thiện.
