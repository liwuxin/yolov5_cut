import cv2
import os
import numpy as np


work_path="D:\GitHub\yolov5"
# 修改当前工作目录
os.chdir(work_path)
cwd = os.getcwd()
print("当前运行目录：", cwd)
# 读取图像
img1 = cv2.imread('2023-03-16_11-37-20truck.jpg')

img2 = cv2.imread('2023-03-16_11-37-21truck.jpg')


# 计算帧差
diff = cv2.absdiff(img1, img2)

# 显示帧差图像
cv2.imshow('diff', diff)
cv2.waitKey(0)
cv2.destroyAllWindows()