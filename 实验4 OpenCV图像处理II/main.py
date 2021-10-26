import cv2 as cv
import numpy as np


# （一）使用OpenCV对图像进行缩放、旋转、相似变换、仿射变换
def demo1():
    img = cv.imread('../lena.bmp')
    cv.imshow('origin', img)
    # 按照指定的宽度、高度缩放图片
    res = cv.resize(img, (200, 200))
    # 按照比例缩放，如x,y方向均缩小一倍
    res2 = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    # cv.imshow('res', res)
    # cv.imshow('res2', res2)

    # 旋转
    rows, cols = img.shape[:2]
    # 逆时针45°旋转图片并缩小一半，第一个参数为旋转中心
    M = cv.getRotationMatrix2D((cols / 2, rows / 2), 45, 0.5)
    # img：源图像；M：旋转仿射矩阵；(cols,rows)：dst的大小
    dst = cv.warpAffine(img, M, (cols, rows))
    # cv.imshow('rotation', dst)

    # 相似变换 -- 不是很清楚和了解
    rows, cols, ch = img.shape
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    img = cv.warpAffine(img, M, (cols, rows))
    # Similarity Transform（相似变换） = Rotation（旋转） + Translation（平移） + Scale（放缩）
    # 得到相似变换的矩阵  # center：旋转中心 angle：旋转角度   scale：缩放比例  ,感觉跟旋转没啥区别
    M = cv.getRotationMatrix2D(center=(rows / 2, cols / 2),
                               angle=30,
                               scale=0.5)
    # 原图像按照相似矩阵进行相似变换  三个参数：原图像，相似矩阵，画布面积
    img_rotate = cv.warpAffine(img, M, (rows, cols))
    cv.imshow('img_rotate', img_rotate)

    # 仿射变换
    # ● 性质：Parallel lines are still parallel lines（不再具有保角性，具有保平行性）
    # ● 三个非共线的点对（6 parameters）确定一个仿射变换。
    pts1 = np.float32([[0, 0], [cols, 0], [0, rows]])
    pts2 = np.float32([[cols * 0.3, rows * 0.3], [cols * 0.8, rows * 0.2], [cols * 0.1, rows * 0.9]])
    M = cv.getAffineTransform(pts1, pts2)
    dst = cv.warpAffine(img, M, (cols, rows))
    # cv.imshow('dst', dst)

    cv.waitKey(0)
    cv.destroyAllWindows()


# （二）使用OpenCV对图像进行二值化，对比阈值为128和大津法阈值效果
"""
简单的阈值化:
cv2.threshold第一个参数是源图像，它应该是灰度图像. 第二个参数是用于对像素值进行分类的阈值, 
第三个参数是maxVal，它表示如果像素值大于（有时小于）阈值则要给出的值. 
大津法阈值:
根据双峰图像的图像直方图自动计算阈值。
"""


def demo2():
    img = cv.imread('../demo1.bmp', 0)
    img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_CUBIC)
    cv.imshow('origin', img)
    t1, thd = cv.threshold(img, 128, 255, cv.THRESH_BINARY)
    cv.imshow('thd', thd)

    t2, otsu = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)  # Otsu's thresholding
    cv.imshow('otsu', otsu)

    cv.waitKey(0)
    cv.destroyAllWindows()


# （三）使用OpenCV对二值图进行腐蚀、膨胀、开运算、闭运算
"""
https://blog.csdn.net/Vermont_/article/details/108424547
腐蚀:
内核可以在图像中滑动(在二维卷积中)。只有当内核下的所有像素都为1时，
原始图像中的像素(要么为1，要么为0)才会被认为是1，否则会被侵蚀(变成0)。
这样的结果就是，取决于内核的大小，边界附近的所有像素都会被丢弃。
因此，前景对象的厚度或大小会减少，或者只是简单地让图像中的白色区域减少。
它对于去除小的白色噪音，分离两个连接的物体等都很有用。

膨胀：
膨胀操作正好与腐蚀相反。这里，如果内核下至少有一个像素是“1”，那么该像素元素就是“1”。
因此，它增加了图像中的白色区域，或则说增加了前景目标对象的尺寸大小。
通常情况下，在去除噪声以后，在腐蚀操作之后就是膨胀。因为，腐蚀消除了白色的噪音，
但它也缩小了我们的前景物体，所以我们需要扩大回它。因为当噪音消失了，原本应该存在的白色面积也不会自主回来。
而且膨胀在连接物体的破碎部分时也很有用。

开运算：
先进行腐蚀，再进行膨胀就叫做开运算。 通常情况下，含有噪声的图像二值化后，得到的边界是不平滑的，
物体区域具有一些错判的孔洞，背景区域散布着一些小的噪声物体。
对一个图像先进行腐蚀运算然后再膨胀的操作过程称为开运算，
它可以消除细小的物体、在纤细点处分离物体、平滑较大物体的边界时不明显的改变其面积。

闭运算：
先膨胀再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上面的小黑点。
"""


def demo3():
    img = cv.imread('../img_j.png', 0)
    img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    cv.imshow('origin', img)
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv.erode(img, kernel, iterations=1)  # 腐蚀：它对于去除小的白色噪音，分离两个连接的物体等都很有用
    cv.imshow('fushi', erosion)

    erosion = cv.dilate(img, kernel, iterations=1)  # 膨胀： 增加了前景目标对象的尺寸大小
    cv.imshow('pengzhang', erosion)

    kernel = np.ones((10, 10), np.uint8)
    img = cv.imread('../img_j_1.png', 0)
    img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    cv.imshow('origin1', img)
    opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)  # 开运算：先进行腐蚀，再进行膨胀就叫做开运算。
    cv.imshow('opening', opening)

    img = cv.imread('../img_j_2.png', 0)
    img = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    cv.imshow('origin2', img)
    closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)  # 闭运算：先膨胀再腐蚀。它经常被用来填充前景物体中的小洞，或者前景物体上面的小黑点。
    cv.imshow('closing', closing)

    cv.waitKey(0)
    cv.destroyAllWindows()


"""
腐蚀：将图像的边界点消除，使图像沿着边界向内收缩，也可以将小于指定结构体元素的部分去除。它对于去除小的白色噪音，分离两个连接的物体等都很有用。
膨胀：对图像的边界进行扩张。膨胀操作对填补图像分割后图像内所存在的空白相当有帮助。
开运算：对一个图像先进行腐蚀运算然后再膨胀的操作过程称为开运算，它可以消除细小的物体、在纤细点处分离物体、平滑较大物体的边界时不明显的改变其面积。
闭运算: 闭运算是先膨胀、后腐蚀的运算。它有助于关闭前景物体内部的小孔，或去除物体上的小黑点，还可以将不同的前景图像进行连接。
"""


if __name__ == '__main__':
    demo1()
    # demo2()
    # demo3()
