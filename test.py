import torch
import cv2
import os
import sys
import traceback
import datetime

arg={
    "work_path":"D:\GitHub\yolov5",
    "mode_path":"D:/GitHub/yolov5/yolov5x6.pt",
    "device_type":"cuda",
    #"rtsp_path":"rtsp://admin:jjhb123456@113.26.250.153:8001/Streaming/Channels/101",
    "rtsp_path":"20230313_20230313112927_20230313113114_112928.mp4",
    "cut_label":["truck","bus","train"],
    "cut_threshold":0.8,
    "cut_up_pos":0.5,
    "cut_down_pos":0.1,
}

try:
  
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
    cap.set(cv2.CAP_PROP_FPS, 10)
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

                # 加载图像
                ROI_img=frame.copy()
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
                        py0=ROI_img.shape[0]*(1-arg["cut_up_pos"])
                        px1=ROI_img.shape[1]-1
                        py1=ROI_img.shape[0]*(1-arg["cut_down_pos"])
                        cv2.line(ROI_img,(px0,py0),(px1,py1),(0,0,255),5)
                        #筛选图片保存
                        now_name=name_dict[int(rect[5])]
                        if now_name in arg["cut_label"] and rect[4]>arg["cut_threshold"]:#筛选标签和置信度阈值
                            if y2>ROI_img.shape[0]*(1-arg["cut_up_pos"]) and y2<ROI_img.shape[0]*(1-arg["cut_down_pos"]):#图片右下角在设定区域内
                                cv2.imwrite( datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+now_name+".jpg",frame)
                        #在图像上绘制文字
                        text = name_dict[int(rect[5])]+str(rect[4])[:4] # 要绘制的文字
                        org = (int(rect[0]),int(rect[1])+10) # 文字左下角坐标
                        cv2.putText(ROI_img,text , org, font, font_scale, color, thickness)
                cv2.imshow("show",ROI_img)
                cv2.waitKey(1)

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