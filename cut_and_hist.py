import cv2
import numpy as np

def process():
    # 加载图像
    origin_img1 = cv2.imread(r'.\imgs\0610.bmp', 0)  # 作为模板
    origin_img2 = cv2.imread(r'.\imgs\0599.bmp', 0)  # 作为要扣的图

    print(origin_img1.shape)
    img1 = origin_img1[50:1100, 480:1630]
    img2 = origin_img2[50:1100, 480:1630]

    equ_res1 = cv2.equalizeHist(img1)
    equ_res2 = cv2.equalizeHist(img2)

    sub_img = cv2.absdiff(equ_res2, equ_res1)  # 第二个图减第一个图
    ret, mask_front = cv2.threshold(sub_img, 15, 255, cv2.THRESH_BINARY)
    res = cv2.bitwise_and(img2, img2, mask=mask_front)

    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)
    cv2.imshow('res', res)
    cv2.imshow('hist_res', equ_res1)
    cv2.imwrite(r'.\res\4_cut_hist_img1.bmp', equ_res1)
    cv2.imwrite(r'.\res\4_cut_hist_img2.bmp', equ_res2)
    cv2.imwrite(r'.\res\4_cut_hist_res.bmp', res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process()