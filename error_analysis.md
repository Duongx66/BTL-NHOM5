# Error Analysis

## Summary

Mô hình đạt độ chính xác `0.8868` và macro F1 `0.8670` trên tập test.

## Các lớp dễ nhầm

- `glass` → `plastic`: 7 lần
- `plastic` → `glass`: 6 lần
- `cardboard` → `paper`: 4 lần
- `trash` → `paper`: 4 lần

## Nhận xét chi tiết

- `trash` có recall thấp nhất (`0.6190`), có thể do số lượng ảnh nhỏ và sự đa dạng trong dạng vật thể.
- `plastic` và `glass` thường nhầm lẫn khi ảnh có bóng hoặc vật liệu xuyên thấu.
- `cardboard` và `paper` có texture, màu sắc tương tự nên mô hình dễ dự đoán sai.

## Hướng cải thiện

1. Tăng cường dữ liệu cho lớp `trash` bằng augmentation hoặc khai thác thêm ảnh.
2. Thử weighted sampler / class weights cho lớp mất cân bằng.
3. Sử dụng Grad-CAM để kiểm tra xem mô hình có tập trung đúng vùng ảnh không.
4. Xem xét fine-tuning thêm sau khi export ONNX để ổn định mô hình.
