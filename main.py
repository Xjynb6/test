import cv2
import numpy as np
from PIL import Image
import os
import pytesseract

def main(path):
    image_dir = path
    # 遍历目录中的所有文件
    for filename in os.listdir(image_dir):
        # 检查文件是否为图片（这里假设是.jpg或.png格式，你可以根据需要添加更多格式）
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # 构建文件的完整路径
            img_path = os.path.join(image_dir, filename)

            # 读取图片
            frame = cv2.imread(img_path)

            # 检查图片是否成功读取（可能由于文件损坏等原因读取失败）
            if frame is not None:
                ball_color = 'green'
                color_dist = {'red': {'Lower': np.array([0, 60, 60]), 'Upper': np.array([6, 255, 255])},
                              'blue': {'Lower': np.array([100, 80, 46]), 'Upper': np.array([124, 255, 255])},
                              'green': {'Lower': np.array([35, 43, 35]), 'Upper': np.array([90, 255, 255])},
                              }
                gs_frame = cv2.GaussianBlur(frame, (5, 5), 0)  # 高斯模糊
                hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像

                erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细

                inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])

                cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]#获取图片中的轮廓
                c = max(cnts, key=cv2.contourArea) #在边界中找出面积最大的区域
                rect = cv2.minAreaRect(c)    # 绘制出该区域的最小外接矩形
                box = cv2.boxPoints(rect)   # 记录该矩形四个点的位置坐标
                box = np.int0(box)   #将坐标转化为整数

                x, y, w, h = cv2.boundingRect(box) #  获取最小外接轴对齐矩形的坐标

                image = frame[y:y + h, x:x + w]  #获取roi区域
                # 显示裁剪后的图像
                cv2.imshow('Cropped Image', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                #膨胀
                kernel = np.ones((5, 5), np.uint8)
                image = cv2.erode(image, kernel)
                # 灰度图像
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # 二值化
                ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
                # 开操作
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 2))
                bin1 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
                kernel = cv2.getStructuringElement(cv2.MORPH_OPEN, (2, 3))
                bin2 = cv2.morphologyEx(bin1, cv2.MORPH_OPEN, kernel)
                # 让背景为白色  字体为黑  便于识别
                cv2.bitwise_not(bin2, bin2)
                # 识别
                test_message = Image.fromarray(bin2)
                text = pytesseract.image_to_string(test_message,lang='eng')
                # print(f'识别结果：{text}')
                print("识别结果：",text.strip())  # 使用strip()函数去掉末尾残留的奇怪字符


if __name__=="__main__":
    path=r"C:\Users\xjy\Desktop\hanjia\tu"
    main(path)
