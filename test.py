from asyncio.windows_events import NULL
import torch
import cv2
import os
import sys
import traceback
import datetime
import seaborn
import yaml
import json
import threading
import time
import pandas._libs.tslibs.base
img_in=0
det_falge=False
# 打开JSON文件
#with open('D:\GitHub\yolov5\sys.conf', 'r') as f:
with open('./sys.conf', 'r') as f:
    # 读取文件内容并解析为Python字典
    arg = json.load(f)

#arg["rtsp_path"]="20230313_20230313112927_20230313113114_112928.mp4"
def img_det():
        global img_in
        global det_falge
        old_ROI_img=NULL
        while True:
            if type(img_in) != type(0):
                det_falge=True
                # 加载图像
                ROI_img=img_in.copy()
                ROI_img=ROI_img[y:y+h,x:x+w]
                img = ROI_img[..., ::-1]  # OpenCV image (BGR to RGB)
                # Inference
                results = model(img, size=640) # batch of images

                # Results
                results.print()
                name_dict=results.names
                rect_list=results.xyxy[0].tolist()

                font = cv2.FONT_HERSHEY_SIMPLEX  # 字体
                font_scale = 0.5 # 字体大小
                color = (0, 0, 255)  # 字体颜色（BGR格式）
                thickness = 2  # 字体线条粗细
                for rect in rect_list:
                    if len(rect)== 6:
                        #绘制矩形框
                        x1=int(rect[0])
                        y1=int(rect[1])
                        x2=int(rect[2])
                        y2=int(rect[3])
                        cv2.rectangle(ROI_img,(x1,y1),(x2,y2),(0, 255, 0), 2)#颜色为绿色，线宽为 2
                        # 绘制上下限制线
                        print(ROI_img.shape)
                        px0=0
                        py0=int(ROI_img.shape[0]*(1-arg["cut_up_pos"]))
                        px1=int(ROI_img.shape[1]-1)
                        py1=int(ROI_img.shape[0]*(1-arg["cut_down_pos"]))
                        # cv2.line(ROI_img,(px0,py0),(px1,py1),(0,0,255),5)
                        #筛选图片保存
                        #在图像上绘制文字
                        text = name_dict[int(rect[5])]+str(rect[4])[:4] # 要绘制的文字
                        org = (int(rect[0]),int(rect[1])+10) # 文字左下角坐标
                        cv2.putText(ROI_img,text , org, font, font_scale, color, thickness)
                        now_name=name_dict[int(rect[5])]
                        if now_name in arg["cut_label"] and rect[4]>arg["cut_threshold"]:#筛选标签和置信度阈值
                            if y2>ROI_img.shape[0]*(1-arg["cut_up_pos"]) and y2<ROI_img.shape[0]*(1-arg["cut_down_pos"]):#图片右下角在设定区域内
                                if type(old_ROI_img)==type(ROI_img):
                                    # 进行帧差运算
                                    diff = cv2.absdiff(ROI_img, old_ROI_img)
                                    # 计算帧差值
                                    diff_sum = diff.sum()
                                else:
                                    diff_sum=arg["diff_threshold"]+1
                                if diff_sum > arg["diff_threshold"]:
                                    cv2.imwrite( arg["cut_savePath"]+"/"+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+now_name+".jpg",frame)
                                    old_ROI_img=ROI_img.copy()
                                    break#一张图片只存一次
                cv2.imshow("show",ROI_img)
                key = cv2.waitKey(1)
                if key == 27:
                    break
                det_falge=False
                img_in=0
            time.sleep(0.01)
try:
    img_thread = threading.Thread(target=img_det).start()

    # 修改当前工作目录
    os.chdir(arg["work_path"])
    cwd = os.getcwd()
    print("当前运行目录：", cwd)

    # Model
    model = torch.hub.load('./', 'custom', path=arg["mode_path"],source='local')
    device = torch.device(arg["device_type"])  # 指定使用 GPU 进行计算
    model.to(device)

    # 创建 VideoCapture 对象，打开 MP4 视频文件
    cap = cv2.VideoCapture(arg["rtsp_path"])
    # 设定帧率
    cap.set(cv2.CAP_PROP_FPS, 2)

    # 检查是否成功打开视频
    if not cap.isOpened():
        print("无法打开视频文件")

    roi_flage=True

    # 循环读取视频帧
    while True:
            # 读取视频帧
            ret, frame = cap.read()

            # 如果读取失败，退出循环
            if not ret:
                print("读取失败")
                cap = cv2.VideoCapture(arg["rtsp_path"])#再次尝试
                ret, frame = cap.read()
                # break
            else:
                if roi_flage :
                    # 显示图像，并让用户框选ROI
                    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
                    cv2.destroyAllWindows()
                    # 提取ROI
                    y=int(bbox[1])
                    h=int(bbox[3])
                    x=int(bbox[0])
                    w=int(bbox[2])
                    roi_flage=False
            if det_falge == False:
                det_falge=True
                img_in=frame
            # results.xyxy[0]  # im1 predictions (tensor)
            # results.pandas().xyxy[0]  # im1 predictions (pandas)
            # 检查是否按下了ESC键
            key = cv2.waitKey(10)
            if key == 27:
                break
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("正常退出")
except Exception as e:
    exc_type, exc_value, exc_traceback = sys.exc_info()
    print("异常类型:", exc_type)
    print("异常信息:", exc_value)
    print("异常回溯信息:")
    traceback.print_tb(exc_traceback)