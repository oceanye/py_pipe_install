# -*- coding: utf-8 -*-
 
import numpy as np
import cv2
import camera_config
import random
import math
 
 
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(2, 480)  
 
 
# 鼠标回调函数
def onmouse_pick_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        threeD = param
        distance = math.sqrt( threeD[y][x][0] **2 + threeD[y][x][1] **2 + threeD[y][x][2] **2 ) 
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "m")
 
WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME,  cv2.WINDOW_AUTOSIZE)
 
while True:
  ret, frame = cap.read()
  frame1 = frame[0:480, 0:640]
  frame2 = frame[0:640, 640:1280]  #割开双目图像
 
  imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)  # 将BGR格式转换成灰度图片
  imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
 
  # cv2.remap 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
  # 依据MATLAB测量数据重建无畸变图片
  img1_rectified = cv2.remap(imgL, camera_config.left_map1, camera_config.left_map2, cv2.INTER_LINEAR)
  img2_rectified = cv2.remap(imgR, camera_config.right_map1, camera_config.right_map2, cv2.INTER_LINEAR)  
 
  imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)  
  imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)
 
  # BM 核心算法
  numberOfDisparities = ((640 // 8) + 15) & -16  # 640对应是分辨率的宽
 
  stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)  #立体匹配
  stereo.setROI1(camera_config.validPixROI1)
  stereo.setROI2(camera_config.validPixROI2)
  stereo.setPreFilterCap(31)
  stereo.setBlockSize(15)
  stereo.setMinDisparity(4)
  stereo.setNumDisparities(numberOfDisparities)
  stereo.setTextureThreshold(50)
  stereo.setUniquenessRatio(15)
  stereo.setSpeckleWindowSize(100)
  stereo.setSpeckleRange(32)
  stereo.setDisp12MaxDiff(1)
 
  disparity = stereo.compute(img1_rectified, img2_rectified) # 计算视差
 
  disp = cv2.normalize(disparity, disparity, alpha=100, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  #归一化函数算法
 
  threeD = cv2.reprojectImageTo3D(disparity, camera_config.Q, handleMissingValues=True)  #计算三维坐标数据值
  threeD = threeD * 16 
 
  cv2.setMouseCallback(WIN_NAME, onmouse_pick_points, threeD)
 
  cv2.imshow("left", frame1)
  cv2.imshow("right", frame2)
  # cv2.imshow("left_r", imgL)
  # cv2.imshow("right_r", imgR)
  cv2.imshow(WIN_NAME, disp) 
 
  key = cv2.waitKey(1)
  if key == ord("q"):
    break
 
cap.release()
cv2.destroyAllWindows()