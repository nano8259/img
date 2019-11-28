import cv2
import numpy as np
import matplotlib.pyplot as plt

def process():
    img = cv2.imread(r'.\imgs\0510.bmp', 0)
    gray = img[50:1100, 480:1630]
    gray = cv2.GaussianBlur(gray, (15, 15), 0)
    circles1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,
                                80, param1=14, param2=35, minRadius=480, maxRadius=520)
    circles = circles1[0, :, :]
    circles = np.uint16(np.around(circles))
    for i in circles[:]:
        cv2.circle(gray, (i[0], i[1]), i[2], (255, 0, 0), 5)
        cv2.circle(gray, (i[0], i[1]), 2, (255, 0, 255), 10)
        # cv2.rectangle(img, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 5)

    print("圆心坐标", i[0], i[1])
    cv2.imwrite(r'.\res\7_hough_4.bmp', gray)

if __name__ == "__main__":
    process()
