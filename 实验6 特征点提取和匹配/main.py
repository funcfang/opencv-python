import cv2
import numpy as np

# 使用OpenCV对图像进行Harris，SIFT特征点提取，并标注特征点
"""
https://www.cnblogs.com/wj-1314/p/13364875.html
Harris角点是特征点检测的基础，提出了应用邻域像素点灰度差值概念，
从而进行判断是否为角点，边缘，平滑区域。Harris角点检测原理是利用移动的窗口在图像中计算灰度变化值，
其中关键流程包括转化为灰度图像，计算差分图像，高斯平滑，计算局部极值，确认角点。

SIFT算法的实质是：“不同的尺度空间上查找关键点(特征点)，并计算出关键点的方向” ，
SIFT所查找到的关键点是一些十分突出，不会因光照，仿射变换和噪音等因素而变化的点，如角点、边缘点、暗区的亮点及亮区的暗点等。
"""


def demo1():
    img = cv2.imread('./blox.png')
    origin = img.copy()

    # Harris特征点提取 https://www.cnblogs.com/DOMLX/p/8763369.html
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成灰度图
    # gray = np.float32(gray)
    # 输入图像必须是 float32 ,最后一个参数在 0.04 到 0.05 之间
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow('origin', origin)
    cv2.imshow('Harris', img)
    cv2.waitKey(0)

    # SIFT特征检测
    sift = cv2.xfeatures2d.SIFT_create()  # 得到特征点
    kp = sift.detect(gray, None)
    cv2.drawKeypoints(gray, kp, img)
    cv2.imshow('origin', origin)
    cv2.imshow('SIFT', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()


def demo2():
    img = cv2.imread('./blox.png')
    origin = img.copy()


if __name__ == '__main__':
    demo1()
