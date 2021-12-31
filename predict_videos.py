import cv2
import numpy as np
from PIL import Image
from utils.utils_deeplab import DeepLabV3

video_path = 'G:\DataSet\TownCentreXVID.avi'  # 测试视频路径  
cap = cv2.VideoCapture(video_path)

deeplab = DeepLabV3()

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(np.uint8(frame))
    frame = np.array(deeplab.segmentate_image(frame))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()