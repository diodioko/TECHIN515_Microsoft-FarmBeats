import cv2
import numpy as np
import os

# 定義照片路徑
photo_path = "lolr(112).jpeg"

# 讀取圖像
image = cv2.imread(photo_path)

# 將圖像轉換為HSV色彩空間
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定義顏色閾值，這取決於銹病對葉片的影響，可以進行調整
lower_color = np.array([0, 150, 150])
upper_color = np.array([30, 255, 255])

# 透過設置適當的閾值範圍來找到銹病區域
mask = cv2.inRange(hsv_image, lower_color, upper_color)

# 進行輪廓檢測
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 繪製輪廓
for contour in contours:
    # 忽略過小的輪廓
    if cv2.contourArea(contour) > 100:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, 'Rust', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 顯示結果
cv2.imshow('Result', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
