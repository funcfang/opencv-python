import numpy as np
import cv2


# (一）参考教材18.1.3读取视频文件，完成帧间差分并显示结果
class demo1():

    def run(self, Videopath='./video.avi'):
        frames = self.Video_to_image(Videopath)
        self.absdiff_(frames)

    def Video_to_image(self, Videopath):
        capture = cv2.VideoCapture(Videopath)
        # 得到整个视频的帧数
        framesNum = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        print("frames=", framesNum)
        frames = []

        for i in range(int(framesNum) - 1):
            ret, frame = capture.read()
            frames.append(frame)
        return frames

    def absdiff_(self, frames):
        c_frames = []
        for i in range(len(frames) - 2):
            frame_front = frames[i]
            frame_later = frames[i + 1]
            # 帧间做差
            d_frame = cv2.absdiff(frame_front, frame_later)
            c_frames.append(d_frame)
            cv2.imshow('d_frame', d_frame)
            cv2.waitKey()

        return c_frames


# （二）实现统计均值背景建模和高斯混合建模，并对比背景图
"""
平均背景建模:
一种简单，计算速度快但是对环境光照变化和背景的多模态性比较敏感的一种备件建模算法。
基本思想：
计算每个像素的平均值作为它的背景建模。检测当前帧时，只需要将当前帧像素值I(x,y)减去背景模型中相同位置像素的平均值u(x,y)，
得到差值d(x,y)，将d(x,y)与一个阈值TH进行比较，大于阈值的就认为是前景，否则为背景。输图像为二值图像。
https://blog.csdn.net/FPGATOM/article/details/84202213

混合高斯模型：
在进行前景检测前，先对背景进行训练，对图像中每个背景采用一个混合高斯模型进行模拟，每个背景的混合高斯的个数可以自适应。
然后在测试阶段，对新来的像素进行GMM匹配，如果该像素值能够匹配其中一个高斯，则认为是背景，否则认为是前景。
由于整个过程GMM模型在不断更新学习中，所以对动态背景有一定的鲁棒性。最后通过对一个有树枝摇摆的动态背景进行前景检测，取得了较好的效果。
https://www.cnblogs.com/Asp1rant/p/15177506.html
"""


class demo2():
    def __init__(self, Videopath='./counting_test.avi'):
        self.capture = cv2.VideoCapture(Videopath)

    # 没做出来
    def AvgBack(self):
        cap = self.capture
        # 得到整个视频的帧数
        framesNum = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        print("frames=", framesNum)
        r, frame_ = cap.read()
        temp_total = np.zeros(frame_.shape, np.float)

        for i in range(1, int(framesNum) - 1):
            ret, frame = cap.read()
            temp_total = cv2.accumulate(frame, temp_total)
            # cv2.imshow('frame', frame)
            # print(i, temp_total)

        frame_avg = temp_total / int(framesNum)

        # print(frame_avg)
        cap = self.capture
        print("frames=", framesNum)
        for i in range(1, int(framesNum) - 1):
            ret, frame = cap.read()
            cv2.imshow('test', cv2.absdiff(frame, frame_avg))
            print(i, frame)
            cv2.waitKey()

    def Gaussian(self, drawContours=False, drawRectangle=True):
        cap = self.capture
        # 创建形态学操作时需要使用的核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        # 创建混合高斯模型
        fgbg = cv2.createBackgroundSubtractorMOG2()
        # 将行人在视频中实时标记出
        while (True):
            ret, frame = cap.read()
            fgmask = fgbg.apply(frame)
            # 形态学开运算去噪点
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            # 寻找视频中的轮廓
            im, contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if drawContours:
                n = len(contours)
                for i in range(n):
                    temp = np.zeros(frame.shape, np.uint8)
                    temp = cv2.drawContours(temp, contours, i, (255, 255, 255), 2)
                    cv2.imshow('frame', frame)
                    cv2.imshow("contours", temp)
                cv2.waitKey()

            if drawRectangle:
                for c in contours:
                    # 计算各轮廓的周长
                    perimeter = cv2.arcLength(c, True)
                    if perimeter > 188:
                        # 找到一个直矩形（不会旋转）
                        x, y, w, h = cv2.boundingRect(c)
                        # 画出这个矩形
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.imshow('frame', frame)
                cv2.imshow('fgmask', fgmask)
                k = cv2.waitKey()
                if k == 27:
                    break

        cap.release()
        cv2.destroyAllWindows()


# （四）查阅资料，基于OpenCV实现KCF目标跟踪
"""
代码参考：
https://www.cnblogs.com/annie22wang/p/9366610.html
https://blog.csdn.net/sements/article/details/100586299
KCF认识:
https://blog.csdn.net/u010440456/article/details/81239221
"""


def demo3():
    # 初始化视频捕获设备
    # gVideoDevice = cv2.VideoCapture("./video.avi")
    gVideoDevice = cv2.VideoCapture(0)
    if not gVideoDevice.isOpened():
        print('open video failed')
        return
    else:
        print('open video succeeded')

    # 选择 框选帧
    print("按 enter 选择当前帧，否则继续下一帧")
    while True:
        gCapStatus, gFrame = gVideoDevice.read()
        cv2.imshow("pick frame", gFrame)
        k = cv2.waitKey()
        if k == 13:
            break

    # 框选感兴趣区域
    cv2.destroyWindow("pick frame")
    gROI = cv2.selectROI("ROI frame", gFrame, False)
    if (not gROI):
        print("空框选，退出")
        quit()

    # 初始化追踪器
    gTracker = cv2.TrackerKCF_create()
    gTracker.init(gFrame, gROI)

    # 循环帧读取，开始跟踪
    while True:
        gCapStatus, gFrame = gVideoDevice.read()
        if (gCapStatus):
            # 展示跟踪图片
            status, coord = gTracker.update(gFrame)
            if status:
                message = {"coord": [((int(coord[0]), int(coord[1])),
                                      (int(coord[0] + coord[2]), int(coord[1] + coord[3])))]}
                p1 = (int(coord[0]), int(coord[1]))
                p2 = (int(coord[0] + coord[2]), int(coord[1] + coord[3]))
                cv2.rectangle(gFrame, p1, p2, (255, 0, 0), 2, 1)
                message['msg'] = "is tracking"
            else:
                message['msg'] = "KCF error,需要重新使用初始ROI开始"
            cv2.imshow('tracked image', gFrame)
            print(message)
            key = cv2.waitKey(1)
            if key == 27:
                break
        else:
            print("捕获帧失败")
            quit()


class MessageItem(object):
    # 用于封装信息的类,包含图片和其他信息
    def __init__(self, frame, message):
        self._frame = frame
        self._message = message

    def getFrame(self):
        # 图片信息
        return self._frame

    def getMessage(self):
        # 文字信息,json格式
        return self._message


class Tracker(object):
    '''
    追踪者模块,用于追踪指定目标
    '''

    def __init__(self, tracker_type="BOOSTING", draw_coord=True):
        '''
        初始化追踪器种类
        '''
        # 获得opencv版本
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_type
        self.isWorking = False
        self.draw_coord = draw_coord
        # 构造追踪器
        if int(major_ver) < 3:
            self.tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()

    def initWorking(self, frame, box):
        '''
        追踪器工作初始化
        frame:初始化追踪画面
        box:追踪的区域
        '''
        if not self.tracker:
            raise Exception("追踪器未初始化")
        status = self.tracker.init(frame, box)
        if not status:
            raise Exception("追踪器工作初始化失败")
        self.coord = box
        self.isWorking = True

    def track(self, frame):
        '''
        开启追踪
        '''
        message = None
        if self.isWorking:
            status, self.coord = self.tracker.update(frame)
            if status:
                message = {"coord": [((int(self.coord[0]), int(self.coord[1])),
                                      (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3])))]}
                if self.draw_coord:
                    p1 = (int(self.coord[0]), int(self.coord[1]))
                    p2 = (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3]))
                    cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    message['msg'] = "is tracking"
        return MessageItem(frame, message)


def TrackerDemo():
    # 初始化视频捕获设备
    # gVideoDevice = cv2.VideoCapture("./video.avi")
    gVideoDevice = cv2.VideoCapture(0)
    if not gVideoDevice.isOpened():
        print('open video failed')
        return
    else:
        print('open video succeeded')

    # 选择 框选帧
    print("按 enter 为当前帧，否则继续下一帧")
    while True:
        gCapStatus, gFrame = gVideoDevice.read()
        cv2.imshow("pick frame", gFrame)
        k = cv2.waitKey()
        if k == 13:
            break

    # 框选感兴趣区域region of interest
    cv2.destroyWindow("pick frame")
    gROI = cv2.selectROI("ROI frame", gFrame, False)
    if (not gROI):
        print("空框选，退出")
        quit()

    # 初始化追踪器
    gTracker = Tracker(tracker_type="KCF")
    gTracker.initWorking(gFrame, gROI)

    # 循环帧读取，开始跟踪
    while True:
        gCapStatus, gFrame = gVideoDevice.read()
        if (gCapStatus):
            # 展示跟踪图片
            _item = gTracker.track(gFrame)
            cv2.imshow("track result", _item.getFrame())

            if _item.getMessage():
                # 打印跟踪数据
                print(_item.getMessage())
            else:
                # 丢失，重新用初始ROI初始
                print("丢失，重新使用初始ROI开始")
                gTracker = Tracker(tracker_type="KCF")
                gTracker.initWorking(gFrame, gROI)

            _key = cv2.waitKey(1) & 0xFF
            if (_key == ord('q')) | (_key == 27):
                break
            if (_key == ord('r')):
                # 用户请求用初始ROI
                print("用户请求用初始ROI")
                gTracker = Tracker(tracker_type="KCF")
                gTracker.initWorking(gFrame, gROI)

        else:
            print("捕获帧失败")
            quit()


if __name__ == "__main__":
    # de1 = demo1()
    # de1.run()
    de2 = demo2()
    de2.AvgBack()
    # de2.Gaussian()
    # de2.Gaussian(drawContours=True, drawRectangle=False)
    # demo3()
    # TrackerDemo()
