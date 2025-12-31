"""
Script 1: Export YOLOv8 to ONNX format
Mục đích: Xuất mô hình YOLOv8 sang định dạng ONNX để có thể sử dụng với ONNX Runtime và TensorRT
"""

import os
from ultralytics import YOLO
import onnx


def export_yolov8_to_onnx(model_name: str = 'yolov8n.pt', 
                         imgsz: int = 640,
                         simplify: bool = True):
    """
    Xuất mô hình YOLOv8 sang ONNX
    
    Args:
        model_name: Tên model YOLOv8 (n/s/m/l/x)
        imgsz: Kích thước input image
        simplify: Có đơn giản hóa mô hình ONNX không
    """
    print("=" * 60)
    print("XUẤT MÔ HÌNH YOLOV8 SANG ONNX")
    print("=" * 60)
    
    # Tạo thư mục models nếu chưa tồn tại
    os.makedirs('models', exist_ok=True)
    
    # Load YOLOv8 model
    print(f"\n1. Đang load mô hình YOLOv8: {model_name}")
    model = YOLO(model_name)
    print(f"   ✓ Đã load mô hình thành công!")
    
    # Export to ONNX
    print(f"\n2. Đang xuất mô hình sang ONNX...")
    print(f"   - Input size: {imgsz}x{imgsz}")
    print(f"   - Simplify: {simplify}")
    
    onnx_path = model.export(
        format='onnx',
        imgsz=imgsz,
        simplify=simplify,
        dynamic=False,  # Static shape for TensorRT optimization
        opset=12
    )
    
    print(f"   ✓ Đã xuất ONNX thành công!")
    print(f"   ✓ Đường dẫn: {onnx_path}")
    
    # Validate ONNX model
    print(f"\n3. Đang kiểm tra mô hình ONNX...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"   ✓ Mô hình ONNX hợp lệ!")
    
    # Print model information
    print(f"\n4. Thông tin mô hình ONNX:")
    print(f"   - IR Version: {onnx_model.ir_version}")
    print(f"   - Producer: {onnx_model.producer_name}")
    print(f"   - Opset: {onnx_model.opset_import[0].version}")
    
    # Print input/output information
    print(f"\n   Input:")
    for input_tensor in onnx_model.graph.input:
        dims = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"     - Name: {input_tensor.name}")
        print(f"     - Shape: {dims}")
        print(f"     - Type: {input_tensor.type.tensor_type.elem_type}")
    
    print(f"\n   Output:")
    for output_tensor in onnx_model.graph.output:
        dims = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"     - Name: {output_tensor.name}")
        print(f"     - Shape: {dims}")
        print(f"     - Type: {output_tensor.type.tensor_type.elem_type}")
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT!")
    print(f"Mô hình ONNX đã được lưu tại: {onnx_path}")
    print("=" * 60)
    
    return onnx_path


if __name__ == "__main__":
    # Có thể thử các model size khác nhau:
    # yolov8n.pt - Nano (nhanh nhất, nhỏ nhất)
    # yolov8s.pt - Small
    # yolov8m.pt - Medium
    # yolov8l.pt - Large
    # yolov8x.pt - Extra Large (chính xác nhất, lớn nhất)
    
    model_name = 'yolov8n.pt'  # Bắt đầu với model nhỏ nhất để test
    onnx_path = export_yolov8_to_onnx(model_name=model_name, imgsz=640)
