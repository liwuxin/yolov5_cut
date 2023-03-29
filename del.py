import cv2
import os
import json
import pandas._libs.tslibs.base

os.chdir('D:\GitHub\yolov5\cutImg')

# 读取json文件中的阈值参数
with open('del.json', 'r') as f:
    config = json.load(f)
    threshold = config['threshold']
    delay_ms = config['delay_ms']

# 获取当前目录下所有图片文件
img_list = os.listdir('.')
img_list = [i for i in img_list if i.endswith('.jpg') or i.endswith('.png')]

# 按照文件新建时间排序
img_list.sort(key=lambda x: os.path.getctime(x))

# 读取第一张图片
prev_img = cv2.imread(img_list[0])

# 遍历所有图片
for img_path in img_list[1:]:
    # 读取当前图片
    curr_img = cv2.imread(img_path)

    # 进行帧差运算
    diff = cv2.absdiff(curr_img, prev_img)

    # 计算帧差值
    diff_sum = diff.sum()

    # 显示帧差图像
    # 在帧差图像上添加帧差值信息
    # 显示帧差图像
    if diff_sum < threshold:
        cv2.putText(diff, f"Diff Sum: {diff_sum}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        cv2.putText(diff, f"Diff Sum: {diff_sum}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Frame Difference', diff)
    cv2.waitKey(delay_ms)
    cv2.destroyAllWindows()
    
    # 如果帧差值低于设定阈值，删除当前图片
    if diff_sum < threshold:
        os.remove(img_path)

    # 更新前一张图片
    prev_img = curr_img
