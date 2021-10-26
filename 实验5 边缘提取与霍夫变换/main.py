import cv2
import numpy as np

# （一）使用OpenCV对图像计算梯度，分别使用Sobel和Laplacian算子
"""
https://www.cnblogs.com/wj-1314/p/9800272.html
一阶微分算子:

Sobel算子:
Sobel算子是一种用于边缘检测的离散微分算子，它结合了高斯平滑和微分求导。该算子用于计算图像明暗程度近似值。
根据图像边缘旁边明暗程度把该区域内超过某个数的特定点记为边缘。Sobel
算子在Prewitt算子的基础上增加了权重的概念，认为相邻点的距离远近对当前像素点的影响是不同的，
距离越近的像素点对应当前像素的影响越大，从而实现图像锐化并突出边缘轮廓。
Sobel算子的边缘定位更准确，常用于噪声较多，灰度渐变的图像。

拉普拉斯（Laplacian）算子是n维欧几里德空间中的一个二阶微分算子，常用于图像增强领域和边缘提取。
它通过灰度差分计算邻域内的像素，基本流程是：判断图像中心像素灰度值与它周围其他像素的灰度值，
如果中心像素的灰度更高，则提升中心像素的灰度；反之降低中心像素的灰度，从而实现图像锐化操作。
在算法实现过程中，Laplacian算子通过对邻域中心像素的四方向或八方向求梯度，再将梯度相加起来判断中心像素灰度与邻域内其他像素灰度的关系，
最后通过梯度运算的结果对像素灰度进行调整。
"""


def demo1():
    img = cv2.imread("../demo.jpg", 0)
    img = cv2.resize(img, (300, 300))
    # Sobel算子
    cv2.imshow('origin', img)
    x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
    """
    Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。
    因此要使用16位有符号的数据类型，即cv2.CV_16S。
    在经过处理后，别忘了用convertScaleAbs()函数将其转回原来的uint8形式。否则将无法显示图像，而只是一副灰色的窗口。
    """
    absX = cv2.convertScaleAbs(x)  # 转回uint8
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  # 由于Sobel算子是在两个方向计算的，最后还需要用cv2.addWeighted(...)函数将其组合起来
    # cv2.imshow("absX", absX)
    # cv2.imshow("absY", absY)
    # cv2.imshow("Result", dst)

    # Laplacian算子
    """
    Laplacian算子的基本流程是：判断图像中心像素灰度值与它周围其他像素的灰度值，如果中心像素的灰度更高，则提升中心像素的灰度；
    反之降低中心像素的灰度，从而实现图像锐化操作。
    """
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # 灰度化处理图像
    # grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.imread('../demo.jpg', cv2.IMREAD_GRAYSCALE)
    # 拉普拉斯算法
    dst = cv2.Laplacian(grayImage, cv2.CV_16S, ksize=3)
    Laplacian = cv2.convertScaleAbs(dst)
    cv2.imshow("Laplacian", Laplacian)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# （二）使用OpenCV对图像Canny边缘检测，显示并保存
"""
非微分边缘检测算子——Canny算子:
Canny算法是一种被广泛应用于边缘检测的标准算法，其目标是找到一个最优的边缘检测解或找寻一幅图像中灰度强度变换最强的位置。
最优边缘检测主要通过低错误率，高定位性和最小响应三个标准进行评价。Canny算子的实现步骤如下：
step1: 用高斯滤波器平滑图象；
step2: 计算图像中每个像素点的梯度强度和方向（用一阶偏导的有限差分来计算梯度的幅值和方向）；
step3: 对梯度幅值进行非极大值抑制（Non-Maximum Suppression），以消除边缘检测带来的杂散响应；
step4: 用双阈值算法(Double-Threshold)检测来确定真实和潜在的边缘，通过抑制孤立的弱边缘最终完成边缘检测；
"""


def demo2():
    img = cv2.imread('../demo.jpg')
    img = cv2.resize(img, (300, 300))
    # 原图置灰
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    v2 = cv2.Canny(gray, 50, 100)
    res = np.hstack((gray, v2))
    cv2.imshow('img', res)
    cv2.imwrite('res.bmp', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# （三）使用OpenCV对house.tif进行霍夫直线检测，对硬币图片进行霍夫圆形检测
"""
https://www.freesion.com/article/95061174017/
霍夫变换是用来辨别找出物件中的特征，例如：线条。他的算法流程大致如下，给定一个物件、要辨别的形状的种类，
算法会在参数空间(parameter space)中执行投票来决定物体的形状，而这是由累加空间(accumulator space)里的局部最大值(local maximum)来决定。
"""


def demo3():
    # 霍夫直线检测
    img = cv2.imread('../house.tif')
    cv2.imshow('origin', img)
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    orgb = cv2.cvtColor(gray, cv2.COLOR_BGR2RGB)
    # # 高斯滤波降噪
    gaussian = cv2.GaussianBlur(orgb, (7, 7), 0)
    # # 利用Canny进行边缘检测
    edges = cv2.Canny(gaussian, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)
    for line in lines:
        rho = line[0][0]  # 第一个元素是距离rho
        theta = line[0][1]  # 第二个元素是角度theta
        if (theta < (np.pi / 4.)) or (theta > (3. * np.pi / 4.0)):  # 垂直直线
            pt1 = (int(rho / np.cos(theta)), 0)  # 该直线与第一行的交点
            # 该直线与最后一行的焦点
            pt2 = (int((rho - img.shape[0] * np.sin(theta)) / np.cos(theta)), img.shape[0])
            cv2.line(img, pt1, pt2, (255))  # 绘制一条白线
        else:  # 水平直线
            pt1 = (0, int(rho / np.sin(theta)))  # 该直线与第一列的交点
            # 该直线与最后一列的交点
            pt2 = (img.shape[1], int((rho - img.shape[1] * np.cos(theta)) / np.sin(theta)))
            cv2.line(img, pt1, pt2, (255), 1)
    cv2.imshow("line-HoughLines", img)

    # 自动检测可能的直线，返回的是一条条线段
    # 第二个参数为半径的步长，第三个参数为每次偏转的角度
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=30, maxLineGap=8)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.imshow("line-HoughLinesP", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # 霍夫圆形检测
    img = cv2.imread('../yingbi.png')
    img = cv2.resize(img, (300, 300))
    cv2.imshow('origin', img)
    # 将图像转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯滤波降噪
    gaussian = cv2.GaussianBlur(gray, (7, 7), 0)
    # 利用Canny进行边缘检测
    edges = cv2.Canny(gaussian, 80, 180, apertureSize=3)
    # cv2.imshow("edge",edges)
    # 自动检测圆
    circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1, 20, param1=180, param2=30)
    print(circles)

    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 3)
        cv2.circle(img, (i[0], i[1]), 2, (255, 0, 255), 3)

    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # demo1()
    # demo2()
    demo3()
