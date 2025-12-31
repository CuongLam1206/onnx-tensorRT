"""
Preprocessing and postprocessing utilities for YOLOv8
"""

import cv2
import numpy as np
from typing import Tuple, List


def letterbox(image: np.ndarray, new_shape: Tuple[int, int] = (640, 640), 
              color: Tuple[int, int, int] = (114, 114, 114)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Resize image with unchanged aspect ratio using padding
    
    Args:
        image: Input image (H, W, C)
        new_shape: Target size (height, width)
        color: Padding color
        
    Returns:
        resized_image: Padded and resized image
        ratio: Resize ratio
        padding: (pad_width, pad_height)
    """
    shape = image.shape[:2]  # Current shape [height, width]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    
    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    dw /= 2  # Divide padding into 2 sides
    dh /= 2
    
    if shape[::-1] != new_unpad:  # Resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return image, r, (dw, dh)


def preprocess_image(image_path: str, input_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, np.ndarray, float, Tuple[int, int]]:
    """
    Preprocess image for YOLOv8 inference
    
    Args:
        image_path: Path to input image
        input_size: Model input size (height, width)
        
    Returns:
        preprocessed: Preprocessed image ready for model (1, 3, H, W)
        original_image: Original image
        ratio: Resize ratio
        padding: Padding values
    """
    # Read image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Letterbox resize
    image, ratio, padding = letterbox(original_image, input_size)
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Transpose from HWC to CHW
    image = np.transpose(image, (2, 0, 1))
    
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    
    return image, original_image, ratio, padding


def xywh2xyxy(x: np.ndarray) -> np.ndarray:
    """
    Convert bounding box format from [x_center, y_center, width, height] to [x1, y1, x2, y2]
    
    Args:
        x: Bounding boxes in xywh format
        
    Returns:
        Bounding boxes in xyxy format
    """
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # x1
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # y1
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # x2
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # y2
    return y


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> List[int]:
    """
    Non-Maximum Suppression
    
    Args:
        boxes: Bounding boxes in xyxy format (N, 4)
        scores: Confidence scores (N,)
        iou_threshold: IoU threshold for NMS
        
    Returns:
        List of indices to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def postprocess_detections(output: np.ndarray, original_shape: Tuple[int, int], 
                          ratio: float, padding: Tuple[int, int],
                          conf_threshold: float = 0.25, 
                          iou_threshold: float = 0.45) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Postprocess YOLOv8 output
    
    Args:
        output: Model output (1, 84, 8400) - [batch, features, detections]
                Features: [x, y, w, h, class0_conf, class1_conf, ..., class79_conf]
        original_shape: Original image shape (height, width)
        ratio: Resize ratio from preprocessing
        padding: Padding values from preprocessing
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold for NMS
        
    Returns:
        boxes: Final bounding boxes in xyxy format
        scores: Confidence scores
        class_ids: Class IDs
    """
    # Transpose output from (1, 84, 8400) to (8400, 84)
    predictions = output[0].transpose()
    
    # Extract boxes and scores
    boxes = predictions[:, :4]
    scores = predictions[:, 4:].max(axis=1)
    class_ids = predictions[:, 4:].argmax(axis=1)
    
    # Filter by confidence threshold
    mask = scores > conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    # Convert from xywh to xyxy
    boxes = xywh2xyxy(boxes)
    
    # Scale boxes back to original image
    boxes[:, [0, 2]] -= padding[0]  # Remove x padding
    boxes[:, [1, 3]] -= padding[1]  # Remove y padding
    boxes /= ratio  # Scale back
    
    # Clip boxes to original image boundaries
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])
    
    # Apply NMS
    if len(boxes) > 0:
        keep = nms(boxes, scores, iou_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]
    
    return boxes, scores, class_ids


def draw_detections(image: np.ndarray, boxes: np.ndarray, scores: np.ndarray, 
                    class_ids: np.ndarray, class_names: List[str] = None) -> np.ndarray:
    """
    Draw bounding boxes and labels on image
    
    Args:
        image: Input image
        boxes: Bounding boxes in xyxy format
        scores: Confidence scores
        class_ids: Class IDs
        class_names: List of class names
        
    Returns:
        Image with drawn detections
    """
    image = image.copy()
    
    # COCO class names (YOLOv8 default)
    if class_names is None:
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
    
    # Generate colors for each class
    np.random.seed(42)
    colors = np.random.randint(0, 255, size=(len(class_names), 3), dtype=np.uint8)
    
    for box, score, class_id in zip(boxes, scores, class_ids):
        x1, y1, x2, y2 = box.astype(int)
        color = colors[int(class_id)].tolist()
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        label = f"{class_names[int(class_id)]}: {score:.2f}"
        
        # Draw label background
        (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x1, y1 - label_height - 10), (x1 + label_width, y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return image
