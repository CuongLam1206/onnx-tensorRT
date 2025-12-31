"""
Script 2: ONNX Runtime Inference
Mục đích: Chạy inference với mô hình ONNX sử dụng ONNX Runtime
"""

import os
import time
import cv2
import numpy as np
import onnxruntime as ort
from utils.preprocessing import preprocess_image, postprocess_detections, draw_detections


def run_onnx_inference(onnx_path: str, 
                       image_path: str,
                       conf_threshold: float = 0.25,
                       iou_threshold: float = 0.45,
                       use_gpu: bool = True):
    """
    Chạy inference với ONNX Runtime
    
    Args:
        onnx_path: Đường dẫn đến file ONNX
        image_path: Đường dẫn đến ảnh input
        conf_threshold: Ngưỡng confidence
        iou_threshold: Ngưỡng IoU cho NMS
        use_gpu: Sử dụng GPU hay không
    """
    print("=" * 60)
    print("ONNX RUNTIME INFERENCE")
    print("=" * 60)
    
    # Kiểm tra file tồn tại
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Không tìm thấy file ONNX: {onnx_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    
    # Load ONNX model
    print(f"\n1. Đang load mô hình ONNX...")
    print(f"   - Model: {onnx_path}")
    
    # Chọn execution provider
    if use_gpu:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        print(f"   - Execution Provider: GPU (CUDA) + CPU fallback")
    else:
        providers = ['CPUExecutionProvider']
        print(f"   - Execution Provider: CPU")
    
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Hiển thị thông tin provider đang dùng
    print(f"   ✓ Đã load model thành công!")
    print(f"   ✓ Provider đang dùng: {session.get_providers()[0]}")
    
    # Lấy thông tin input/output
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape = session.get_inputs()[0].shape
    output_shape = session.get_outputs()[0].shape
    
    print(f"\n2. Thông tin mô hình:")
    print(f"   - Input name: {input_name}")
    print(f"   - Input shape: {input_shape}")
    print(f"   - Output name: {output_name}")
    print(f"   - Output shape: {output_shape}")
    
    # Preprocess image
    print(f"\n3. Đang xử lý ảnh input...")
    print(f"   - Image: {image_path}")
    
    preprocessed_img, original_img, ratio, padding = preprocess_image(image_path)
    print(f"   ✓ Đã xử lý ảnh!")
    print(f"   - Original shape: {original_img.shape}")
    print(f"   - Preprocessed shape: {preprocessed_img.shape}")
    print(f"   - Resize ratio: {ratio:.3f}")
    print(f"   - Padding: {padding}")
    
    # Run inference
    print(f"\n4. Đang chạy inference...")
    
    # Warmup
    for _ in range(3):
        _ = session.run([output_name], {input_name: preprocessed_img})
    
    # Benchmark
    num_iterations = 100
    start_time = time.time()
    for _ in range(num_iterations):
        output = session.run([output_name], {input_name: preprocessed_img})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    fps = 1.0 / avg_time
    
    print(f"   ✓ Đã hoàn thành inference!")
    print(f"   - Thời gian trung bình: {avg_time*1000:.2f} ms")
    print(f"   - FPS: {fps:.2f}")
    
    # Postprocess
    print(f"\n5. Đang xử lý kết quả...")
    output = session.run([output_name], {input_name: preprocessed_img})
    
    boxes, scores, class_ids = postprocess_detections(
        output[0],
        original_img.shape[:2],
        ratio,
        padding,
        conf_threshold,
        iou_threshold
    )
    
    print(f"   ✓ Đã xử lý kết quả!")
    print(f"   - Số lượng đối tượng phát hiện: {len(boxes)}")
    
    if len(boxes) > 0:
        print(f"\n6. Chi tiết các đối tượng phát hiện:")
        # COCO class names
        class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        for i, (box, score, class_id) in enumerate(zip(boxes, scores, class_ids)):
            x1, y1, x2, y2 = box.astype(int)
            class_name = class_names[int(class_id)]
            print(f"   {i+1}. {class_name} - Confidence: {score:.3f} - Box: [{x1}, {y1}, {x2}, {y2}]")
    
    # Draw detections
    print(f"\n7. Đang vẽ kết quả...")
    result_img = draw_detections(original_img, boxes, scores, class_ids)
    
    # Save result
    output_path = image_path.replace('.', '_onnx_result.')
    cv2.imwrite(output_path, result_img)
    print(f"   ✓ Đã lưu kết quả tại: {output_path}")
    
    # Display result
    print(f"\n8. Hiển thị kết quả (nhấn phím bất kỳ để đóng)...")
    cv2.imshow('ONNX Runtime - YOLOv8 Detection', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT!")
    print("=" * 60)
    
    return boxes, scores, class_ids, avg_time


if __name__ == "__main__":
    # Đường dẫn
    onnx_path = "yolov8n.onnx"  # Thay đổi nếu model ở vị trí khác
    image_path = "E:\AI\yolov8-onnx-tensorrt\images\sample\sample.jpeg"  # Thay đổi đường dẫn ảnh của bạn
    
    # Chạy inference
    try:
        boxes, scores, class_ids, inference_time = run_onnx_inference(
            onnx_path=onnx_path,
            image_path=image_path,
            conf_threshold=0.25,
            iou_threshold=0.45,
            use_gpu=True  # Đổi thành False nếu không có GPU
        )
    except Exception as e:
        print(f"\nLỗi: {e}")
        print("\nGợi ý:")
        print("1. Đảm bảo đã chạy script 1_export_onnx.py trước")
        print("2. Kiểm tra đường dẫn file ONNX và ảnh")
        print("3. Nếu chạy GPU, đảm bảo đã cài onnxruntime-gpu và CUDA")
