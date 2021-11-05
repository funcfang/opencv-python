import time

import cv2
import numpy as np
import imutils

# （一） 使用OpenCV对图像进行Harris，SIFT特征点提取，并标注特征点
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


# （二）使用OpenCV生成特征的SIFT描述子，对两幅有重叠的图片进行描述子匹配
def demo2():
    img_left = cv2.imread('./left.png')
    img_right = cv2.imread('./right.png')
    img_left = cv2.resize(img_left, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)  # 缩放
    img_right = cv2.resize(img_right, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_CUBIC)

    # 创建SIFT特征检测器
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img_left, None)
    kp2, des2 = sift.detectAndCompute(img_right, None)

    # 暴力匹配
    bf = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
    matches = bf.match(des1, des2)

    # 绘制匹配
    matches = sorted(matches, key=lambda x: x.distance)
    result = cv2.drawMatches(img_left, kp1, img_right, kp2, matches[:100], None)
    cv2.imshow("match", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# （三）使用OpenCV对两幅有重叠的图片匹配后进行拼接，生成全景图（选做）
"""
https://blog.csdn.net/qq_43697752/article/details/107807251
"""

"""
https://www.cnblogs.com/my-love-is-python/p/9755456.html
"""


class Stitcher:
    # 拼接函数
    def stitch(self, images, ratio=0.75, reprojThresh=4.0, showMatches=False):
        # 获取输入图片
        imageB, imageA = images
        # 检测A，B图片的SIFT关键特征点， 并计算特征描述子
        (kpsA, featuresA) = self.detectAndDescribe(imageA)
        (kpsB, featuresB) = self.detectAndDescribe(imageB)
        # 匹配两种图片的所有特征点，并返回结果
        M = self.matchKeypoints(kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh)

        # 如果返回结果为空， 没有匹配成功的特征点，退出算法
        if M is None:
            return None

        # 否则，提取匹配结果
        # H是3*3视角变换矩阵
        (matches, H, status) = M
        # 将图片A进行视角变换， result是变化后图片
        result = cv2.warpPerspective(imageA, H, (imageA.shape[1] + imageB.shape[1], imageA.shape[0]))
        # 将B图片传入result图片最左端
        result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB

        # 检验是否需要显示图片匹配
        if showMatches:
            # 生成匹配图片
            vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches, status)
            # 返回结果
            return (result, vis)

    def detectAndDescribe(self, image):
        # 将彩色图片转换为灰度图
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 建立SIFT生成器
        descriptor = cv2.xfeatures2d.SIFT_create()
        # 检测SIFT特征点，并计算描述子
        (kps, features) = descriptor.detectAndCompute(image, None)

        # 将结果转换为Numpy数组
        kps = np.float32([kp.pt for kp in kps])
        # 返回特征点集， 及对应的描述特征
        return (kps, features)

    def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB, ratio, reprojThresh):
        # 建立暴力匹配器
        matcher = cv2.DescriptorMatcher_create('BruteForce')

        # 使用KNN检测来自A，B图的SIFT特征匹配对， K=2
        rawMatches = matcher.knnMatch(featuresA, featuresB, 2)

        matches = []
        for m in rawMatches:
            # 当最近距离跟次近距离的比值小于ratio值时，保留此匹配对
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                # 储存两个点在featuresA， featuresB中的索引值
                matches.append((m[0].trainIdx, m[0].queryIdx))

        # 当筛选后的匹配对大于4时， 计算视角变化矩阵
        if len(matches) > 4:
            # 获取匹配对的点坐标
            ptsA = np.float32([kpsA[i] for (_, i) in matches])
            ptsB = np.float32([kpsB[i] for (i, _) in matches])

            # 计算视角变化矩阵
            (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)

            # 返回结果
            return (matches, H, status)

        return None

    def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
        # 初始化可视化图片， 将A，B图左右连接
        (hA, wA) = imageA.shape[:2]
        (hB, wB) = imageB.shape[:2]
        vis = np.zeros((max(hA, hB), wA + wB, 3), dtype='uint8')
        vis[0:hA, 0:wA] = imageA
        vis[0:hB, wA:] = imageB

        # 联合遍历， 画出匹配对
        for ((trainIdx, queryIdx), s) in zip(matches, status):
            # 当点对匹配成功时，画到可视化图上

            if s == 1:
                # 画出匹配对
                ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
                ptB = (int(kpsB[trainIdx][0] + wA), int(kpsB[trainIdx][1]))
                cv2.line(vis, ptA, ptB, (0, 255, 0), 1)

        # 返回可视化结果
        return vis


def demo3():
    # 读取拼接图片
    # imageA = cv2.imread("./left.png")
    # imageB = cv2.imread("./right.png")
    imageA = cv2.imread("./left_1.jpg")
    imageB = cv2.imread("./right_1.jpg")
    imageA = cv2.resize(imageA, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)  # 缩放
    imageB = cv2.resize(imageB, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # 对图像进行拼接
    stitcher = Stitcher()  # 对类进行实例化
    (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
    cv2.imshow('vis.jpg', vis)
    cv2.imshow('result.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def demo3():
#     # 读取拼接图片
#     imageA = cv2.imread("./left.png")
#     imageB = cv2.imread("./right.png")
#     imageA = cv2.resize(imageA, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)  # 缩放
#     imageB = cv2.resize(imageB, None, fx=0.7, fy=0.7, interpolation=cv2.INTER_CUBIC)
#
#     stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
#
#     _result, pano = stitcher.stitch((imageA, imageB))
#     cv2.imshow("match", pano)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
