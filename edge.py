import cv2
import numpy as np

def process():
    # 加载图像
    origin_img1 = cv2.imread(r'.\imgs\0610.bmp', 0)  # 作为模板
    origin_img2 = cv2.imread(r'.\imgs\0538.bmp', 0)  # 作为要扣的图
    img1_l = cv2.Laplacian(origin_img1, cv2.CV_8U)
    img1_sx = cv2.Sobel(origin_img1, cv2.CV_8U, 0, 1)
    img1_sy = cv2.Sobel(origin_img1, cv2.CV_8U, 1, 0)
    edges = cv2.Canny(origin_img2, 0, 100)  # canny边缘检测

    sobelX = np.uint8(np.absolute(img1_sx))  # x方向梯度的绝对值
    sobelY = np.uint8(np.absolute(img1_sy))  # y方向梯度的绝对值

    sobelCombined_ = cv2.bitwise_or(img1_sx, img1_sy)
    sobelCombined = cv2.bitwise_or(sobelX, sobelY)

    ret, res = cv2.threshold(sobelCombined, 20, 255, cv2.THRESH_BINARY)
    ret, res_ = cv2.threshold(sobelCombined_, 20, 255, cv2.THRESH_BINARY)


    cv2.imwrite(r'.\res\6_edge_Laplacian.bmp', img1_l)
    cv2.imwrite(r'.\res\6_edge_Sobel.bmp', sobelCombined)
    cv2.imwrite(r'.\res\6_edge_Sobel_threshold.bmp', res)
    cv2.imwrite(r'.\res\6_edge_Sobel_threshold_noabs.bmp', res_)
    cv2.imwrite(r'.\res\6_edge_Sobel_threshold_canny.bmp', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    process()