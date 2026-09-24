import cv2
import numpy as np
import time
import math

# 左相机内参
left_camera_matrix = np.array([[416.841180253704, 0.0, 338.485167779639],
                                         [0., 416.465934495134, 230.419201769346],
                                         [0., 0., 1.]])
 
# 左相机畸变系数:[k1, k2, p1, p2, k3]
left_distortion = np.array([[-0.0170280933781798, 0.0643596519467521, -0.00161785356900972, -0.00330684695473645, 0]])
 
# 右相机内参
right_camera_matrix = np.array([[417.765094485395, 0.0, 315.061245379892],
                                          [0., 417.845058291483, 238.181766936442],
                                            [0., 0., 1.]])
# 右相机畸变系数:[k1, k2, p1, p2, k3]                                          
right_distortion = np.array([[-0.0394089328586398, 0.131112076868352, -0.00133793245429668, -0.00188957913931929, 0]])

# 旋转矩阵
R = np.array([[0.999962872853149, 0.00187779299260463, -0.00840992323112715],
                           [ -0.0018408858041373, 0.999988651353238, 0.00439412154902114],
                           [ 0.00841807904053251, -0.00437847669953504, 0.999954981430194]])
 
# 平移向量
T = np.array([[-120.326603502087], [0.199732192805711], [-0.203594457929446]])

# 图像尺寸
size = (640, 480)

# 立体校正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R, T)

# 校正查找映射表
left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

# 初始化视频捕获
capture = cv2.VideoCapture(1)
WIN_NAME = 'Deep disp'
cv2.namedWindow(WIN_NAME, cv2.WINDOW_AUTOSIZE)
imageWidth = 640
imageHeight = 480
print("Setting camera resolution to:", imageWidth * 2, "x", imageHeight)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth * 2)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight)

actual_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Actual resolution set:", actual_width, "x", actual_height)

if not capture.isOpened():
    print("Error: Could not open video capture.")
    exit()

while True:
    t1 = time.time()
    ret, frame = capture.read()
    if not ret:
        print("Captured no image")
        continue
    frame1 = frame[0:480, 0:640]
    frame2 = frame[0:640, 640:1280]
    imgL = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

    imageL = cv2.cvtColor(img1_rectified, cv2.COLOR_GRAY2BGR)
    imageR = cv2.cvtColor(img2_rectified, cv2.COLOR_GRAY2BGR)

    blockSize = 8
    img_channels = 3
    stereo = cv2.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,  
                                   preFilterCap=140,
                                   uniquenessRatio=1,      
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=cv2.STEREO_SGBM_MODE_HH)
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化函数算法，生成深度图（灰度图）
    disp = cv2.normalize(disparity, disparity, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 生成深度图（颜色图）
    dis_color = cv2.applyColorMap(cv2.convertScaleAbs(disparity, alpha=255/16), cv2.COLORMAP_JET)

    threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    threeD = threeD * 16

    cv2.imshow("depth", dis_color)

    cv2.imshow("left", imageL)
    cv2.imshow("right", imageR)
    cv2.imshow(WIN_NAME, disp)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()