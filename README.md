# YOLOv8 với ONNX & TensorRT

Dự án học tập về ONNX và TensorRT thông qua việc triển khai YOLOv8 object detection.

## 📋 Mục lục

- [Giới thiệu](#giới-thiệu)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Cấu trúc dự án](#cấu-trúc-dự-án)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
- [So sánh hiệu suất](#so-sánh-hiệu-suất)
- [Troubleshooting](#troubleshooting)

## 🎯 Giới thiệu

Project này giúp bạn:
- Hiểu cách xuất mô hình PyTorch sang ONNX
- Chạy inference với ONNX Runtime
- Tối ưu hóa mô hình với TensorRT
- So sánh hiệu suất giữa các framework

**Các bước thực hiện:**
1. Export YOLOv8 → ONNX
2. Inference với ONNX Runtime
3. Convert ONNX → TensorRT Engine
4. Inference với TensorRT (GPU)
5. So sánh performance

## 💻 Yêu cầu hệ thống

### Cho ONNX Runtime (CPU/GPU):
- Python 3.8+
- Windows/Linux/MacOS
- (Tùy chọn) NVIDIA GPU với CUDA 11.x hoặc 12.x

### Cho TensorRT (chỉ GPU):
- NVIDIA GPU (Compute Capability ≥ 6.0)
- CUDA Toolkit 11.x hoặc 12.x
- cuDNN 8.x
- TensorRT 8.x hoặc 10.x
- Linux hoặc Windows

## 📦 Cài đặt

### Bước 1: Clone hoặc tạo project

```bash
cd e:\AI\yolov8-onnx-tensorrt
```

### Bước 2: Tạo virtual environment (khuyến nghị)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies

#### Cài đặt cơ bản (CPU + ONNX Runtime):
```bash
pip install -r requirements.txt
```

#### Cài đặt cho GPU (ONNX Runtime GPU):
```bash
pip install onnxruntime-gpu  # Thay vì onnxruntime
```

#### Cài đặt TensorRT (GPU only):

**Linux:**
```bash
pip install tensorrt
pip install pycuda
```

**Windows:**
1. Tải TensorRT từ [NVIDIA Developer](https://developer.nvidia.com/tensorrt)
2. Giải nén và thêm vào PATH
3. Cài đặt Python wheel:
   ```bash
   pip install tensorrt-10.x.x-cp3x-none-win_amd64.whl
   pip install pycuda
   ```

### Bước 4: Chuẩn bị ảnh test

Tạo thư mục `images` và đặt ảnh test vào đó:

```bash
mkdir images
# Copy ảnh của bạn vào images/sample.jpg
```

## 📁 Cấu trúc dự án

```
yolov8-onnx-tensorrt/
├── utils/                      # Utility functions
│   ├── __init__.py
│   └── preprocessing.py        # Tiền xử lý và hậu xử lý
├── images/                     # Thư mục chứa ảnh test
│   └── sample.jpg
├── models/                     # Thư mục lưu models (tự động tạo)
├── 1_export_onnx.py           # Script 1: Export YOLOv8 → ONNX
├── 2_onnx_inference.py        # Script 2: Inference với ONNX Runtime
├── 3_tensorrt_convert.py      # Script 3: Convert ONNX → TensorRT
├── 4_tensorrt_inference.py    # Script 4: Inference với TensorRT
├── demo.py                    # Demo so sánh ONNX vs TensorRT
├── requirements.txt           # Python dependencies
└── README.md                  # File này
```

## 🚀 Hướng dẫn sử dụng

### Script 1: Export ONNX

Xuất mô hình YOLOv8 sang định dạng ONNX:

```bash
python 1_export_onnx.py
```

**Output:** `yolov8n.onnx` (khoảng 6MB)

**Các model size khác:**
- `yolov8n.pt` - Nano (nhanh nhất, 6MB)
- `yolov8s.pt` - Small (9MB)
- `yolov8m.pt` - Medium (26MB)
- `yolov8l.pt` - Large (44MB)
- `yolov8x.pt` - Extra Large (68MB, chính xác nhất)

### Script 2: ONNX Runtime Inference

Chạy inference với ONNX Runtime:

```bash
python 2_onnx_inference.py
```

**Output:** 
- Kết quả detection trên terminal
- Ảnh với bounding boxes: `images/sample_onnx_result.jpg`
- Hiển thị ảnh kết quả

**Tùy chỉnh:**
```python
# Trong file 2_onnx_inference.py
onnx_path = "yolov8n.onnx"
image_path = "images/your_image.jpg"
use_gpu = True  # False nếu chỉ dùng CPU
```

### Script 3: Convert sang TensorRT

Chuyển đổi ONNX sang TensorRT engine (yêu cầu GPU):

```bash
python 3_tensorrt_convert.py
```

**Output:** `yolov8n_fp16.engine`

**Các precision mode:**
- `fp32` - Độ chính xác cao nhất, chậm nhất
- `fp16` - Cân bằng (khuyến nghị) - nhanh hơn ~2x
- `int8` - Nhanh nhất, cần calibration

**Tùy chỉnh:**
```python
# Trong file 3_tensorrt_convert.py
precision='fp16'  # Đổi thành 'fp32' hoặc 'int8'
max_workspace_size=2  # GB, tăng nếu có nhiều RAM
```

### Script 4: TensorRT Inference

Chạy inference với TensorRT (yêu cầu GPU):

```bash
python 4_tensorrt_inference.py
```

**Output:**
- Kết quả detection trên terminal
- Ảnh với bounding boxes: `images/sample_tensorrt_result.jpg`
- Thời gian inference (ms) và FPS

### Demo: So sánh ONNX vs TensorRT

Chạy cả hai và so sánh hiệu suất:

```bash
python demo.py
```

**Output:**
- So sánh side-by-side ONNX vs TensorRT
- Metrics: inference time, FPS, speedup
- Ảnh so sánh: `comparison_result.jpg`

## 📊 So sánh hiệu suất

### Kết quả thực tế trên NVIDIA RTX 4050

Benchmark với YOLOv8n, input size 640x640 (GPU vs GPU):

| Framework | Device | Precision | Inference Time | FPS | Speedup |
|-----------|--------|-----------|----------------|-----|---------|
| ONNX Runtime | GPU (RTX 4050) | FP32 | 7.90 ms | 126.57 | 1x (baseline) |
| TensorRT | GPU (RTX 4050) | FP16 | 2.27 ms | 439.72 | **3.47x** 🚀 |

### Chi tiết kết quả:

**ONNX Runtime (GPU RTX 4050 với CUDA 12.x):**
- ⏱️ Thời gian: 7.90 ms
- 📊 FPS: 126.57
- 🎯 Detections: 2 objects
- ✅ Chạy trên GPU với CUDA ExecutionProvider

**TensorRT (GPU RTX 4050):**
- ⚡ Thời gian: 2.27 ms
- 🚀 FPS: 439.72 (gần 440 FPS!)
- 🎯 Detections: 2 objects
- 💚 Tiết kiệm thời gian: 71.2%

### So sánh tổng quan:

**TensorRT vs ONNX Runtime (GPU):** Nhanh hơn **3.47x**  
**TensorRT vs CPU:** Nhanh hơn **~16x** (440 FPS vs 27 FPS)  
**ONNX GPU vs CPU:** Nhanh hơn **~4.7x** (127 FPS vs 27 FPS)

**Lưu ý:** 
- Cần cài CUDA Toolkit 12.x để ONNX Runtime chạy trên GPU
- TensorRT tối ưu hơn cho inference trên NVIDIA GPU
- Hiệu suất phụ thuộc vào GPU, model size, input size

### Ưu điểm từng framework:

**ONNX Runtime:**
- ✅ Cross-platform (CPU/GPU/Mobile)
- ✅ Dễ sử dụng
- ✅ Không cần setup phức tạp
- ❌ Chậm hơn TensorRT

**TensorRT:**
- ✅ Cực kỳ nhanh trên GPU NVIDIA
- ✅ Tối ưu hóa tự động
- ✅ Hỗ trợ FP16, INT8
- ❌ Chỉ chạy trên NVIDIA GPU
- ❌ Setup phức tạp hơn

## 🔧 Troubleshooting

### Lỗi: "No module named 'ultralytics'"
```bash
pip install ultralytics
```

### Lỗi: "No module named 'onnxruntime'"
```bash
pip install onnxruntime-gpu  # hoặc onnxruntime cho CPU
```

### Lỗi: "No module named 'tensorrt'"
- Đảm bảo đã cài TensorRT đúng cách
- Kiểm tra CUDA và cuDNN
- Xem hướng dẫn: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/

### Lỗi: "CUDA out of memory"
- Giảm `max_workspace_size` trong script 3
- Sử dụng ảnh nhỏ hơn
- Đóng các ứng dụng khác đang dùng GPU

### ONNX Runtime chạy chậm trên GPU
- Kiểm tra provider: `print(ort.get_device())`
- Cài đặt `onnxruntime-gpu` thay vì `onnxruntime`
- Kiểm tra CUDA: `nvidia-smi`

### TensorRT build engine lâu
- Bình thường, lần đầu build có thể mất 2-5 phút
- Engine được lưu lại, lần sau load nhanh
- Tăng `verbose=True` để xem tiến trình

## 📚 Tài liệu tham khảo

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ONNX GitHub](https://github.com/onnx/onnx)

## 📝 License

MIT License - Tự do sử dụng cho mục đích học tập và thương mại.

## 🤝 Đóng góp

Mọi đóng góp đều được hoan nghênh! Tạo issue hoặc pull request nếu bạn có cải tiến.

---

**Happy Learning! 🚀**

Nếu gặp vấn đề, hãy kiểm tra phần [Troubleshooting](#troubleshooting) hoặc tạo issue.
