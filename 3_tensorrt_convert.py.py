"""
Script 3: Convert ONNX to TensorRT Engine
Mục đích: Chuyển đổi mô hình ONNX sang TensorRT engine để tối ưu hóa tốc độ trên GPU NVIDIA
Lưu ý: Script này yêu cầu GPU NVIDIA và TensorRT đã được cài đặt
"""

import os
import sys


def check_tensorrt():
    """Kiểm tra TensorRT có được cài đặt không"""
    try:
        import tensorrt as trt
        return True, trt.__version__
    except ImportError:
        return False, None


def convert_onnx_to_tensorrt(onnx_path: str,
                             engine_path: str = None,
                             precision: str = 'fp16',
                             max_workspace_size: int = 1,
                             verbose: bool = True):
    """
    Chuyển đổi ONNX sang TensorRT engine
    
    Args:
        onnx_path: Đường dẫn file ONNX
        engine_path: Đường dẫn lưu TensorRT engine (None = tự động tạo)
        precision: Độ chính xác (fp32/fp16/int8)
        max_workspace_size: Kích thước workspace tối đa (GB)
        verbose: Hiển thị chi tiết quá trình
    """
    # Kiểm tra TensorRT
    has_trt, trt_version = check_tensorrt()
    if not has_trt:
        print("=" * 60)
        print("LỖI: TENSORRT CHƯA ĐƯỢC CÀI ĐẶT")
        print("=" * 60)
        print("\nTensorRT không được tìm thấy trên hệ thống!")
        print("\nHướng dẫn cài đặt TensorRT:")
        print("\n1. Đối với Linux:")
        print("   pip install tensorrt")
        print("\n2. Đối với Windows:")
        print("   - Tải TensorRT từ: https://developer.nvidia.com/tensorrt")
        print("   - Giải nén và thêm vào PATH")
        print("   - Cài đặt Python wheel: pip install tensorrt-*-cp3*-*.whl")
        print("\n3. Yêu cầu:")
        print("   - NVIDIA GPU với CUDA support")
        print("   - CUDA Toolkit đã cài đặt")
        print("   - cuDNN đã cài đặt")
        print("=" * 60)
        return None
    
    import tensorrt as trt
    
    print("=" * 60)
    print("CHUYỂN ĐỔI ONNX SANG TENSORRT")
    print("=" * 60)
    
    print(f"\nTensorRT version: {trt_version}")
    
    # Kiểm tra file ONNX
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"Không tìm thấy file ONNX: {onnx_path}")
    
    # Tạo tên file engine
    if engine_path is None:
        engine_path = onnx_path.replace('.onnx', f'_{precision}.engine')
    
    print(f"\n1. Thông tin chuyển đổi:")
    print(f"   - Input ONNX: {onnx_path}")
    print(f"   - Output Engine: {engine_path}")
    print(f"   - Precision: {precision.upper()}")
    print(f"   - Max workspace: {max_workspace_size} GB")
    
    # Khởi tạo TensorRT components
    print(f"\n2. Đang khởi tạo TensorRT builder...")
    
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print(f"   ✓ Đã khởi tạo builder!")
    
    # Parse ONNX
    print(f"\n3. Đang parse file ONNX...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('   ✗ Lỗi khi parse ONNX:')
            for error in range(parser.num_errors):
                print(f"      {parser.get_error(error)}")
            return None
    
    print(f"   ✓ Đã parse ONNX thành công!")
    
    # Build engine config
    print(f"\n4. Đang cấu hình TensorRT engine...")
    config = builder.create_builder_config()
    
    # TensorRT 10.x: Sử dụng set_memory_pool_limit thay vì max_workspace_size
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size * (1 << 30))  # Convert GB to bytes
    
    # Thiết lập precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"   ✓ Đã bật FP16 mode")
        else:
            print(f"   ⚠ GPU không hỗ trợ FP16, sử dụng FP32")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print(f"   ✓ Đã bật INT8 mode")
            print(f"   ⚠ Lưu ý: INT8 cần calibration dataset để đạt độ chính xác tốt")
        else:
            print(f"   ⚠ GPU không hỗ trợ INT8, sử dụng FP32")
    else:
        print(f"   ✓ Sử dụng FP32 mode (mặc định)")
    
    # Build engine
    print(f"\n5. Đang build TensorRT engine...")
    print(f"   ⏳ Quá trình này có thể mất vài phút...")
    
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print(f"   ✗ Lỗi khi build engine!")
        return None
    
    print(f"   ✓ Đã build engine thành công!")
    
    # Save engine
    print(f"\n6. Đang lưu TensorRT engine...")
    with open(engine_path, 'wb') as f:
        f.write(serialized_engine)
    
    file_size = os.path.getsize(engine_path) / (1024 * 1024)  # MB
    print(f"   ✓ Đã lưu engine!")
    print(f"   ✓ Kích thước: {file_size:.2f} MB")
    
    # Hiển thị thông tin engine
    print(f"\n7. Thông tin TensorRT engine:")
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    
    # TensorRT 10.x: Sử dụng num_io_tensors thay vì num_bindings
    print(f"   - Số lượng I/O tensors: {engine.num_io_tensors}")
    for i in range(engine.num_io_tensors):
        tensor_name = engine.get_tensor_name(i)
        tensor_shape = engine.get_tensor_shape(tensor_name)
        tensor_dtype = engine.get_tensor_dtype(tensor_name)
        tensor_mode = engine.get_tensor_mode(tensor_name)
        is_input = (tensor_mode == trt.TensorIOMode.INPUT)
        print(f"   - {'Input' if is_input else 'Output'} {i}: {tensor_name}")
        print(f"     Shape: {tensor_shape}, Dtype: {tensor_dtype}")
    
    print("\n" + "=" * 60)
    print("HOÀN TẤT!")
    print(f"TensorRT engine đã được lưu tại: {engine_path}")
    print("=" * 60)
    
    return engine_path


if __name__ == "__main__":
    # Đường dẫn
    onnx_path = "yolov8n.onnx"
    
    # Các tùy chọn precision:
    # - 'fp32': Độ chính xác cao nhất, chậm nhất
    # - 'fp16': Cân bằng giữa tốc độ và độ chính xác (khuyến nghị)
    # - 'int8': Nhanh nhất, cần calibration
    
    try:
        engine_path = convert_onnx_to_tensorrt(
            onnx_path=onnx_path,
            precision='fp16',  # Thay đổi thành 'fp32' hoặc 'int8' nếu cần
            max_workspace_size=2,  # 2 GB
            verbose=False  # True để xem chi tiết quá trình build
        )
    except Exception as e:
        print(f"\nLỗi: {e}")
        print("\nGợi ý:")
        print("1. Đảm bảo đã cài đặt TensorRT")
        print("2. Kiểm tra GPU NVIDIA và CUDA hoạt động")
        print("3. Kiểm tra đường dẫn file ONNX")
