from __future__ import print_function
import cv2
import matplotlib.pyplot as plt;
import numpy as np


# （一）使用OpenCV进行RGB到HSV和YUV色彩空间转换，并显示保存
def demo1():
    img = cv2.imread("demo.bmp")
    cv2.imshow('before', img)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # HSV分别是色调（Hue），饱和度（Saturation）和明度（Value）
    cv2.imshow('img_hsv', img_hsv)
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)  # Y表示明亮度，U、V表示色度(浓度)；色度信号是由两个互相独立的信号U和V组成
    cv2.imshow('img_yuv', img_yuv)
    cv2.imwrite('img_hsv.bmp', img_hsv)
    cv2.imwrite('img_yuv.bmp', img_yuv)
    cv2.waitKey(0)


# （二）使用OpenCV将彩色图片转成灰度图片，并得到图片的灰度直方图
def demo2():
    img = cv2.imread("demo.bmp")
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imshow('img_gray', img_gray)
    cv2.imwrite('img_gray.bmp', img_gray)
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    show_hist(hist)
    # plt.hist(img_gray.ravel(), 256, [0, 256])  #这种方法也行
    # plt.show()
    cv2.waitKey(0)


def show_hist(hist_):
    plt.bar(list(range(256)), hist_[:, 0])
    plt.xlim([0, 256])
    plt.show()


# （三）使用OpenCV对图片进行点运算，以及gamma矫正。
'''
    Gamma变换就是用来图像增强，其提升了暗部细节，
    简单来说就是通过非线性变换，让图像从暴光强度的线性响应变得更接近人眼感受的响应，
    即将漂白（相机曝光）或过暗（曝光不足）的图片，进行矫正。
    点运算：g(x) = af (x) + b； a>0, a称为增益(gain)，b称为偏差bias, 他们分别控制图片对比度(contrast)和亮度(brightness)
    点运算参考：https://zhuanlan.zhihu.com/p/73694512
    更多可以参考:https://www.cnblogs.com/silence-cho/p/11006958.html
'''


def demo3():
    # 加载原始图像
    img = cv2.imread('duibi_demo.jpg')
    # img = cv2.imread('demo.bmp')
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width / 1.7), int(height / 1.7)))  # 图片缩放
    cv2.imshow("origin", img)
    adjust_point(img, -1, 0)
    # adjust_point(img, 1, 50)
    # adjust_point(img, 1, -50)
    # adjust_point(img, 0.3, 0)
    # adjust_point(img, 1.5, 0)
    # adjust_point(img, 2, 0)
    # adjust_gamma(img, 0.1)
    # adjust_gamma(img, 0.5)
    # adjust_gamma(img, 1.5)
    cv2.waitKey(0)


# 点运算
def adjust_point(imgOri, a, b):
    imgOut = np.zeros(imgOri.shape, imgOri.dtype)
    imgOut = cv2.convertScaleAbs(imgOri, alpha=a, beta=b)  # 最为准确的方法。a不能为负数？
    # 为什么底下4种写法运行结果各有差异
    # imgOut = np.float(a) * imgOri + b
    # imgOut[imgOut > 255] = 255
    # imgOut = np.round(imgOut)
    # imgOut = imgOut.astype(np.uint8)

    # high, wide, channel = imgOri.shape
    # for row in range(high):
    #     for col in range(wide):
    #         b_, g, r = imgOri[row, col]
    #         b_ = a * b_ + b
    #         g = a * g + b
    #         r = a * r + b
    #         imgOut[row, col] = b_, g, r

    # imgOut = a * imgOri + b

    # imgOut = np.uint8(np.clip((a * imgOri + b), 0, 255))
    cv2.putText(imgOut, f"a={a},b={b}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow(f"a={a},b={b}", imgOut)
    cv2.imwrite(f"a={a}_b={b}.bmp", imgOut)
    return imgOut


# 伽马校正
def adjust_gamma(imgOri, v):
    imgOri = imgOri / 255.0  # 注意255.0得采用浮点数
    imgOut = np.power(imgOri, v) * 255.0
    imgOut = imgOut.astype(np.uint8)
    cv2.putText(imgOut, f"v={v}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
    cv2.imshow(f"v={v}", imgOut)
    cv2.imwrite(f"v={v}.bmp", imgOut)
    return imgOut


# 伽马校正另外一种方法
def adjust_ga(image, gamma=1.0):
    # 建立查找表，将像素值[0，255]映射到调整后的伽玛值
    # 遍历[0，255]范围内的所有像素值来构建查找表，然后再提高到反伽马的幂-然后将该值存储在表格中
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")
    # 使用查找表应用伽玛校正
    return cv2.LUT(image, table)


# （四）使用OpenCV对图片进行均值滤波、高斯滤波、中值滤波
# 消除图像中的噪声成分叫作图像的平滑化或滤波操作。
# 参考：https://www.cnblogs.com/wj-1314/p/11693364.html
def demo4():
    img = cv2.imread("noise_demo.png")
    height, width = img.shape[:2]
    img = cv2.resize(img, (int(width / 1.2), int(height / 1.2)))  # 图片缩放
    cv2.imshow('img', img)
    # 均值滤波  简单的平均卷积操作
    img_mean = cv2.blur(img, (5, 5))
    cv2.imshow('img_mean', img_mean)
    cv2.imwrite('img_mean.bmp', img_mean)
    # 高斯滤波 高斯模糊的卷积
    img_Guassian = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imshow('img_Guassian', img_Guassian)
    cv2.imwrite('img_Guassian.bmp', img_Guassian)
    # 中值滤波 利用中值替换
    img_median = cv2.medianBlur(img, 5)
    cv2.imshow('img_median', img_median)
    cv2.imwrite('img_median.bmp', img_median)
    cv2.waitKey(0)


'''
（一）查阅资料，探讨均值滤波、高斯滤波、中值滤波对不同噪声的作用。
均值滤波
均值滤波是典型的线性滤波算法，它是指在图像上对目标像素给一个模板，该模板包括了其周围的临近像素，
再用模板中的全体像素的平均值来代替原来像素值。
均值滤波不能很好地保护图像细节，在图像去噪的同时也破坏了图像的细节，使图像变得模糊，不能很好地去除噪声点，特别是椒盐噪声。

高斯滤波
高斯模糊本质上是低通滤波器，输出图像的每个像素点是原图像上对应像素点与周围像素点的加权和。
高斯滤波一般针对的是高斯噪声，能够很好的抑制图像输入时随机引入的噪声，将像素点跟邻域像素看作是一种高斯分布的关系，
它的操作是将图像和一个高斯核进行卷积操作。

中值滤波
中值滤波是一种典型的非线性滤波技术，基本思想是用像素点邻域灰度值的中值来代替该像素点的灰度值，
即中值滤波将窗口函数里面的所有像素进行排序取得中位数来代表该窗口中心的像素值，对椒盐噪声和脉冲噪声的抑制效果特别好。
'''

if __name__ == '__main__':
# demo1()
# demo2()
# demo3()
# demo4()
