import numpy as np
import cv2

def process():
    img = cv2.imread(r'.\all_res\0505.bmp', 0)
    blur = cv2.GaussianBlur(img, (7, 7), 0)

    equ = cv2.equalizeHist(blur)
    # ret, binary = cv2.threshold(equ, 130, 255, cv2.THRESH_BINARY)
    # ret, binary2 = cv2.threshold(equ, 190, 255, cv2.THRESH_BINARY_INV)
    #
    # # 轮廓检测
    # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # # 绘制轮廓
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    lap = cv2.Laplacian(equ, 0)

    cv2.imshow('equ', equ)
    # cv2.imshow('binary', binary)
    # cv2.imshow('binary2', binary2)
    cv2.imshow('img', lap)
    cv2.waitKey()


if __name__ == "__main__":
    process()
    # water_image()