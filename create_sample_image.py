"""
Tạo ảnh mẫu đơn giản để test YOLOv8
"""
import numpy as np
import cv2
import os

# Tạo thư mục images
os.makedirs('images', exist_ok=True)

# Tạo ảnh với kích thước 640x640
img = np.ones((480, 640, 3), dtype=np.uint8) * 220

# Vẽ tiêu đề
cv2.putText(img, "YOLOv8 Test Image", (150, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 2)

# Vẽ rectangle giống xe
cv2.rectangle(img, (100, 150), (280, 280), (100, 100, 200), -1)
cv2.rectangle(img, (100, 150), (280, 280), (0, 0, 0), 2)
cv2.putText(img, "Car-like", (130, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# Vẽ hình người
cv2.rectangle(img, (350, 140), (420, 300), (50, 150, 50), -1)
cv2.circle(img, (385, 110), 30, (50, 150, 50), -1)
cv2.rectangle(img, (350, 140), (420, 300), (0, 0, 0), 2)
cv2.putText(img, "Person-like", (340, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Vẽ circle
cv2.circle(img, (500, 200), 40, (0, 100, 200), -1)
cv2.circle(img, (500, 200), 40, (0, 0, 0), 2)
cv2.putText(img, "Ball", (470, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Text hướng dẫn
cv2.putText(img, "Hoac thay bang anh that cua ban!", 
            (80, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2)

# Lưu ảnh
cv2.imwrite('images/sample.jpg', img)
print("✓ Đã tạo ảnh mẫu: images/sample.jpg")
print("  Bạn có thể thay bằng ảnh thật để test tốt hơn!")
