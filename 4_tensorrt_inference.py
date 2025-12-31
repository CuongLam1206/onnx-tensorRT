"""
Script 4: TensorRT Inference
Mục đích: Chạy inference với TensorRT engine để đạt tốc độ tối đa trên GPU NVIDIA
Lưu ý: Yêu cầu GPU NVIDIA, CUDA, và TensorRT
"""

import os
import time
import cv2
import numpy as np
from utils.preprocessing import preprocess_image, postprocess_detections, draw_detections


def check_dependencies():
    """Kiểm tra các dependencies cần thiết"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        return True, trt.__version__
    except ImportError as e:
        return False, str(e)


def run_tensorrt_inference(engine_path: str,
                           image_path: str,
                           conf_threshold: float = 0.25,
                           iou_threshold: float = 0.45):
    """
    Chạy inference với TensorRT engine
    
    Args:
        engine_path: Đường dẫn đến TensorRT engine
        image_path: Đường dẫn đến ảnh input
        conf_threshold: Ngưỡng confidence
        iou_threshold: Ngưỡng IoU cho NMS
    """
    # Kiểm tra dependencies
    has_deps, info = check_dependencies()
    if not has_deps:
        print("=" * 60)
        print("LỖI: THIẾU DEPENDENCIES")
        print("=" * 60)
        print(f"\nLỗi: {info}")
        print("\nCần cài đặt:")
        print("1. TensorRT: pip install tensorrt")
        print("2. PyCUDA: pip install pycuda")
        print("\nYêu cầu:")
        print("- NVIDIA GPU")
        print("- CUDA Toolkit")
        print("=" * 60)
        return None
    
    import tensorrt as trt
    import pycuda.driver as cuda
    
    print("=" * 60)
    print("TENSORRT INFERENCE")
    print("=" * 60)
    
    print(f"\nTensorRT version: {info}")
    
    # Kiểm tra file
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Không tìm thấy TensorRT engine: {engine_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    
    # Load TensorRT engine
    print(f"\n1. Đang load TensorRT engine...")
    print(f"   - Engine: {engine_path}")
    
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        serialized_engine = f.read()
    
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    context = engine.create_execution_context()
    
    print(f"   ✓ Đã load engine thành công!")
    
    # TensorRT 10.x: Lấy thông tin tensors
    print(f"\n2. Thông tin engine:")
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        is_input = (tensor_mode == trt.TensorIOMode.INPUT)
        print(f"   - {'Input' if is_input else 'Output'} {i}: {tensor_name}")
        print(f"     Shape: {tensor_shape}, Dtype: {tensor_dtype}")
    
    # Allocate buffers
    print(f"\n3. Đang cấp phát memory buffers...")
    
    # TensorRT 10.x: Get tensor names and shapes
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    # Get shapes
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    # Allocate host and device buffers
    dtype = np.float32
    
    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=dtype)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=dtype)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    print(f"   ✓ Đã cấp phát buffers!")
    print(f"   - Input buffer: {h_input.nbytes / 1024 / 1024:.2f} MB")
    print(f"   - Output buffer: {h_output.nbytes / 1024 / 1024:.2f} MB")
    
    # Preprocess image
    print(f"\n4. Đang xử lý ảnh input...")
    print(f"   - Image: {image_path}")
    
    preprocessed_img, original_img, ratio, padding = preprocess_image(image_path)
    
    print(f"   ✓ Đã xử lý ảnh!")
    print(f"   - Original shape: {original_img.shape}")
    print(f"   - Preprocessed shape: {preprocessed_img.shape}")
    
    # Prepare input
    np.copyto(h_input, preprocessed_img.ravel())
    
    # Run inference
    print(f"\n5. Đang chạy inference với TensorRT...")
    
    # TensorRT 10.x: Set tensor addresses
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    # Warmup
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    # Benchmark
    num_iterations = 100
    start_time = time.time()
    
    for _ in range(num_iterations):
        # Transfer input data to device
        cuda.memcpy_htod_async(d_input, h_input, stream)
        
        # Run inference (TensorRT 10.x: execute_async_v3)
        context.execute_async_v3(stream_handle=stream.handle)
        
        # Transfer predictions back to host
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        
        # Synchronize stream
        stream.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    fps = 1.0 / avg_time
    
    print(f"   ✓ Đã hoàn thành inference!")
    print(f"   - Thời gian trung bình: {avg_time*1000:.2f} ms")
    print(f"   - FPS: {fps:.2f}")
    
    # Reshape output
    output = h_output.reshape(output_shape)
    
    # Postprocess
    print(f"\n6. Đang xử lý kết quả...")
    
    boxes, scores, class_ids = postprocess_detections(
        output,
        original_img.shape[:2],
        ratio,
        padding,
        conf_threshold,
        iou_threshold
    )
    
    print(f"   ✓ Đã xử lý kết quả!")
    print(f"   - Số lượng đối tượng phát hiện: {len(boxes)}")
    
    if len(boxes) > 0:
        print(f"\n7. Chi tiết các đối tượng phát hiện:")
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
    print(f"\n8. Đang vẽ kết quả...")
    result_img = draw_detections(original_img, boxes, scores, class_ids)
    
    # Save result
    output_path = image_path.replace('.', '_tensorrt_result.')
    cv2.imwrite(output_path, result_img)
    print(f"   ✓ Đã lưu kết quả tại: {output_path}")
    
    # Display result
    print(f"\n9. Hiển thị kết quả (nhấn phím bất kỳ để đóng)...")
    cv2.imshow('TensorRT - YOLOv8 Detection', result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT!")
    print("=" * 60)
    
    return boxes, scores, class_ids, avg_time


if __name__ == "__main__":
    # Đường dẫn
    engine_path = "yolov8n_fp16.engine"  # Hoặc yolov8n_fp32.engine
    image_path = "E:\AI\yolov8-onnx-tensorrt\images\sample\sample.jpeg"
    
    # Chạy inference
    try:
        boxes, scores, class_ids, inference_time = run_tensorrt_inference(
            engine_path=engine_path,
            image_path=image_path,
            conf_threshold=0.25,
            iou_threshold=0.45
        )
    except Exception as e:
        print(f"\nLỗi: {e}")
        print("\nGợi ý:")
        print("1. Đảm bảo đã chạy script 3_tensorrt_convert.py trước")
        print("2. Kiểm tra đường dẫn TensorRT engine và ảnh")
        print("3. Đảm bảo đã cài đặt TensorRT và PyCUDA")
        print("4. Kiểm tra GPU và CUDA hoạt động")
