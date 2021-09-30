import cv2
import numpy as np


# （二）使用OpenCV将读取的图片，并将中间1/4面积像素改为0，并且保存图片
def demo1():
    img = cv2.imread("demo.bmp")
    cv2.imshow('before', img)
    height = img.shape[0]
    width = img.shape[1]
    img[int(0.25 * height):int(0.75 * height), int(0.25 * width):int(0.75 * width)] = 0
    cv2.imshow('after', img)
    cv2.waitKey(0)
    cv2.imwrite('after1.bmp', img)


# （三）使用OpenCV在图片中画矩形,圆形,直线,文字，并显示和保存
def demo2():
    img = np.zeros((512, 512, 3), np.uint8)
    cv2.line(img, (230, 155), (280, 155), (255, 0, 0), 4)  # 起点和终点坐标
    cv2.rectangle(img, (165, 65), (345, 245), (0, 255, 0), 3)  # 左上角顶点和右下角顶点坐标
    cv2.circle(img, (255, 155), 50, (0, 0, 300), 3)  # 中心点和半径
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Hello OpenCV  fq', (120, 300), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('after', img)
    cv2.waitKey(0)
    cv2.imwrite('after2.bmp', img)


if __name__ == '__main__':
    # demo1()
    demo2()
