import cv2
import numpy as np

def process():
    # 加载图像
    img1 = cv2.imread(r'.\imgs\0610.bmp')  # 作为模板
    img2 = cv2.imread(r'.\imgs\0599.bmp')  # 作为要扣的图

    sub_img = cv2.absdiff(img2, img1)  # 第二个图减第一个图
    ret, mask_front = cv2.threshold(sub_img, 15, 255, cv2.THRESH_BINARY)
    #res = cv2.bitwise_and(img2, img2, mask=mask_front)

    cv2.rectangle(img1, (480, 50), (1630, 1100), (0, 255, 0), 3)

    cv2.imwrite(r'.\res\rec2.bmp', img1)
    # cv2.imshow('img1', img1)
    # cv2.imshow('img2', img2)
    #cv2.imshow('res', res)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process()