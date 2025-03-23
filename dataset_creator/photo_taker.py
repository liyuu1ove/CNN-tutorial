
import cv2
import time
import os
def set_cam_autowb(cam, enable=True, manual_temp=5500, hue=0):
    """
    设置摄像头自动白平衡
    enable: 是否启用自动白平衡
    manual_temp: 手动模式下的色温
    """
    cam.set(cv2.CAP_PROP_AUTO_WB, int(enable))
    if not enable:
        cam.set(cv2.CAP_PROP_WB_TEMPERATURE, manual_temp)


cap =cv2.VideoCapture(0)
# change_cam_resolution(cap, 640, 480, 60)
# set_manual_exporsure(cap, 350)

#set_cam_autowb(cap, True)

photo_dir = "./raw"
if not os.path.exists(photo_dir):
    os.makedirs(photo_dir)

# cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
photo_start_num =  318   #照片命名起点数字
photo_take_delay = 1    #拍照间隔 
filehead = 'img-'       #照片名前后缀
filetype = '.jpg' 
while cap.isOpened():
    # Read frame from the video
    time.sleep(photo_take_delay)
    ret, frame = cap.read()
    if not ret:
        break
    
    filename = filehead + str(photo_start_num) + filetype
    file_path = os.path.join(photo_dir, filename)
    cv2.imwrite(file_path,frame)
    cv2.imshow('1',frame)
    print(f"img saved!",photo_start_num)
    photo_start_num = photo_start_num + 1
    if photo_start_num == 1:
        break
