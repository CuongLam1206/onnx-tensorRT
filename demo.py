"""
Demo Script: So sánh ONNX Runtime vs TensorRT
Mục đích: Chạy inference với cả ONNX Runtime và TensorRT, so sánh hiệu suất
"""

import os
import time
import cv2
import numpy as np
from typing import Tuple, Optional


def check_onnx_available():
    """Kiểm tra ONNX Runtime có sẵn không"""
    try:
        import onnxruntime as ort
        return True, ort.get_device()
    except ImportError:
        return False, None


def check_tensorrt_available():
    """Kiểm tra TensorRT có sẵn không"""
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        return True, trt.__version__
    except ImportError:
        return False, None


def run_onnx_inference_benchmark(onnx_path: str, 
                                 preprocessed_img: np.ndarray,
                                 num_iterations: int = 100,
                                 use_gpu: bool = True) -> Tuple[np.ndarray, float]:
    """Chạy benchmark với ONNX Runtime"""
    import onnxruntime as ort
    
    # Create session
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)
    
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Warmup
    for _ in range(10):
        _ = session.run([output_name], {input_name: preprocessed_img})
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        output = session.run([output_name], {input_name: preprocessed_img})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    
    # Get final output
    output = session.run([output_name], {input_name: preprocessed_img})
    
    return output[0], avg_time


def run_tensorrt_inference_benchmark(engine_path: str,
                                     preprocessed_img: np.ndarray,
                                     num_iterations: int = 100) -> Tuple[np.ndarray, float]:
    """Chạy benchmark với TensorRT"""
    import tensorrt as trt
    import pycuda.driver as cuda
    
    # Load engine
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(TRT_LOGGER)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate buffers - TensorRT 10.x API
    # Get tensor names
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    
    # Get tensor shapes
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)
    
    # Calculate volume
    def volume(shape):
        vol = 1
        for dim in shape:
            vol *= dim
        return vol
    
    h_input = cuda.pagelocked_empty(volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(volume(output_shape), dtype=np.float32)
    
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    
    stream = cuda.Stream()
    
    # Prepare input
    np.copyto(h_input, preprocessed_img.ravel())
    
    # Set tensor addresses for TensorRT 10.x
    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    
    # Warmup
    for _ in range(10):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        cuda.memcpy_htod_async(d_input, h_input, stream)
        context.execute_async_v3(stream_handle=stream.handle)
        cuda.memcpy_dtoh_async(h_output, d_output, stream)
        stream.synchronize()
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_iterations
    
    output = h_output.reshape(output_shape)
    
    return output, avg_time


def create_comparison_image(onnx_img: np.ndarray, 
                           tensorrt_img: np.ndarray,
                           onnx_time: float,
                           tensorrt_time: float) -> np.ndarray:
    """Tạo ảnh so sánh side-by-side"""
    
    # Resize images to same height if different
    h1, w1 = onnx_img.shape[:2]
    h2, w2 = tensorrt_img.shape[:2]
    
    if h1 != h2:
        target_h = min(h1, h2)
        onnx_img = cv2.resize(onnx_img, (int(w1 * target_h / h1), target_h))
        tensorrt_img = cv2.resize(tensorrt_img, (int(w2 * target_h / h2), target_h))
        h1, w1 = onnx_img.shape[:2]
        h2, w2 = tensorrt_img.shape[:2]
    
    # Create comparison canvas
    gap = 20
    canvas = np.ones((h1 + 100, w1 + w2 + gap, 3), dtype=np.uint8) * 255
    
    # Place images
    canvas[50:50+h1, 0:w1] = onnx_img
    canvas[50:50+h2, w1+gap:w1+gap+w2] = tensorrt_img
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    
    # ONNX label
    onnx_label = f"ONNX Runtime"
    onnx_time_label = f"{onnx_time*1000:.2f} ms ({1.0/onnx_time:.1f} FPS)"
    cv2.putText(canvas, onnx_label, (10, 30), font, font_scale, (0, 0, 255), thickness)
    cv2.putText(canvas, onnx_time_label, (10, h1 + 80), font, font_scale * 0.7, (0, 0, 0), thickness)
    
    # TensorRT label
    trt_label = f"TensorRT"
    trt_time_label = f"{tensorrt_time*1000:.2f} ms ({1.0/tensorrt_time:.1f} FPS)"
    cv2.putText(canvas, trt_label, (w1 + gap + 10, 30), font, font_scale, (0, 128, 255), thickness)
    cv2.putText(canvas, trt_time_label, (w1 + gap + 10, h1 + 80), font, font_scale * 0.7, (0, 0, 0), thickness)
    
    # Speedup
    speedup = onnx_time / tensorrt_time
    speedup_label = f"TensorRT nhanh hon {speedup:.2f}x"
    speedup_color = (0, 200, 0) if speedup > 1 else (0, 0, 200)
    cv2.putText(canvas, speedup_label, (w1 // 2 - 100, h1 + 80), font, font_scale * 0.8, speedup_color, thickness)
    
    return canvas


def main():
    """Main demo function"""
    print("=" * 80)
    print(" " * 20 + "SO SÁNH ONNX RUNTIME vs TENSORRT")
    print("=" * 80)
    
    # Kiểm tra dependencies
    has_onnx, onnx_info = check_onnx_available()
    has_trt, trt_info = check_tensorrt_available()
    
    print(f"\n✓ ONNX Runtime: {'Có sẵn' if has_onnx else 'Không có'}")
    if has_onnx:
        print(f"  Device: {onnx_info}")
    
    print(f"✓ TensorRT: {'Có sẵn' if has_trt else 'Không có'}")
    if has_trt:
        print(f"  Version: {trt_info}")
    
    if not has_onnx:
        print("\n⚠ Cần cài đặt ONNX Runtime: pip install onnxruntime-gpu")
        return
    
    # Đường dẫn
    onnx_path = "yolov8n.onnx"
    tensorrt_path = "yolov8n_fp16.engine"
    # Try multiple possible image paths
    possible_paths = [
        "images/sample.jpg",
        "images/sample/sample.jpeg",
        "images/sample.jpeg"
    ]
    image_path = None
    for path in possible_paths:
        if os.path.exists(path):
            image_path = path
            break
    
    if image_path is None:
        image_path = "images/sample.jpg"  # Default for error message
    
    # Kiểm tra files
    if not os.path.exists(onnx_path):
        print(f"\n✗ Không tìm thấy: {onnx_path}")
        print("  Chạy: python 1_export_onnx.py")
        return
    
    if has_trt and not os.path.exists(tensorrt_path):
        print(f"\n⚠ Không tìm thấy: {tensorrt_path}")
        print("  Chạy: python 3_tensorrt_convert.py")
        has_trt = False
    
    if not os.path.exists(image_path):
        print(f"\n✗ Không tìm thấy ảnh: {image_path}")
        print("  Tạo thư mục 'images' và đặt ảnh test vào đó")
        return
    
    # Import utilities
    from utils.preprocessing import preprocess_image, postprocess_detections, draw_detections
    
    # Preprocess image
    print(f"\n{'='*80}")
    print("1. PREPROCESSING")
    print(f"{'='*80}")
    preprocessed_img, original_img, ratio, padding = preprocess_image(image_path)
    print(f"✓ Shape: {original_img.shape} -> {preprocessed_img.shape}")
    
    # Run ONNX inference
    print(f"\n{'='*80}")
    print("2. ONNX RUNTIME INFERENCE")
    print(f"{'='*80}")
    
    onnx_output, onnx_time = run_onnx_inference_benchmark(
        onnx_path, 
        preprocessed_img,
        num_iterations=100,
        use_gpu=True
    )
    
    print(f"✓ Inference time: {onnx_time*1000:.2f} ms")
    print(f"✓ FPS: {1.0/onnx_time:.2f}")
    
    # Postprocess ONNX results
    onnx_boxes, onnx_scores, onnx_class_ids = postprocess_detections(
        onnx_output,
        original_img.shape[:2],
        ratio,
        padding
    )
    
    print(f"✓ Detections: {len(onnx_boxes)}")
    
    onnx_result_img = draw_detections(original_img.copy(), onnx_boxes, onnx_scores, onnx_class_ids)
    
    # Run TensorRT inference if available
    if has_trt:
        print(f"\n{'='*80}")
        print("3. TENSORRT INFERENCE")
        print(f"{'='*80}")
        
        trt_output, trt_time = run_tensorrt_inference_benchmark(
            tensorrt_path,
            preprocessed_img,
            num_iterations=100
        )
        
        print(f"✓ Inference time: {trt_time*1000:.2f} ms")
        print(f"✓ FPS: {1.0/trt_time:.2f}")
        
        # Postprocess TensorRT results
        trt_boxes, trt_scores, trt_class_ids = postprocess_detections(
            trt_output,
            original_img.shape[:2],
            ratio,
            padding
        )
        
        print(f"✓ Detections: {len(trt_boxes)}")
        
        trt_result_img = draw_detections(original_img.copy(), trt_boxes, trt_scores, trt_class_ids)
        
        # Create comparison
        print(f"\n{'='*80}")
        print("4. SO SÁNH KẾT QUẢ")
        print(f"{'='*80}")
        
        speedup = onnx_time / trt_time
        
        print(f"\nONNX Runtime:")
        print(f"  - Thời gian: {onnx_time*1000:.2f} ms")
        print(f"  - FPS: {1.0/onnx_time:.2f}")
        print(f"  - Detections: {len(onnx_boxes)}")
        
        print(f"\nTensorRT:")
        print(f"  - Thời gian: {trt_time*1000:.2f} ms")
        print(f"  - FPS: {1.0/trt_time:.2f}")
        print(f"  - Detections: {len(trt_boxes)}")
        
        print(f"\nTốc độ tăng: {speedup:.2f}x")
        print(f"Tiết kiệm thời gian: {(1 - trt_time/onnx_time)*100:.1f}%")
        
        # Save and show comparison
        comparison_img = create_comparison_image(onnx_result_img, trt_result_img, onnx_time, trt_time)
        
        output_path = "comparison_result.jpg"
        cv2.imwrite(output_path, comparison_img)
        print(f"\n✓ Đã lưu kết quả so sánh: {output_path}")
        
        cv2.imshow('ONNX vs TensorRT Comparison', comparison_img)
        print(f"\nNhấn phím bất kỳ để đóng...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        # Only show ONNX result
        print(f"\n{'='*80}")
        print("3. KẾT QUẢ (CHỈ ONNX)")
        print(f"{'='*80}")
        
        output_path = "onnx_result.jpg"
        cv2.imwrite(output_path, onnx_result_img)
        print(f"\n✓ Đã lưu kết quả: {output_path}")
        
        cv2.imshow('ONNX Runtime Detection', onnx_result_img)
        print(f"\nNhấn phím bất kỳ để đóng...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\n{'='*80}")
    print("HOÀN TẤT!")
    print(f"{'='*80}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Lỗi: {e}")
        import traceback
        traceback.print_exc()
