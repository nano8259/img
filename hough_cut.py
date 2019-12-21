import cv2
import numpy as np
import os

# root_dir为要读取文件的根目录
root_dir = r".\imgs"

def process(filename):
    open_filename = root_dir + "\\" + filename
    write_filename1 = ".\\hough_light_res_3" + "\\" + filename
    write_filename2 = ".\\hough_light_circle_3" + "\\" + filename
    img = cv2.imread(open_filename, 0)
    gray = img[10:1090, 450:1680]
    # cv2.imwrite(write_filename2 + '_real.bmp', gray)
    gray = cv2.resize(gray, (0, 0), fx=0.905, fy=1)
    light = np.zeros_like(gray)
    light_y, light_x = light.shape
    light[round(light_y/2):light_y, 0:round(light_x / 2)] = 50
    # gray = cv2.add(gray, light)
    detected_edges = cv2.GaussianBlur(gray, (15, 15), 0)
    #detected_edges = cv2.GaussianBlur(detected_edges, (15, 15), 0)
    detected_edges = cv2.add(detected_edges, light)
    detected_edges = cv2.equalizeHist(detected_edges)
    # detected_edges = cv2.GaussianBlur(detected_edges, (15, 15), 0)
    circles1 = cv2.HoughCircles(detected_edges, cv2.HOUGH_GRADIENT, 1,
                                500, param1=80, param2=20, minRadius=480, maxRadius=520)
    circles = circles1[0, :, :]
    circles = np.uint16(np.around(circles))
    min_x, min_y = 9999, 9999
    max_x, max_y = 0, 0
    for i in circles[:]:
        x_p = 566; y_p = 550; r_p = 517
        cv2.circle(detected_edges, (x_p, y_p), r_p, (255, 0, 0), 5)
        cv2.circle(detected_edges, (i[0], i[1]), 2, (255, 0, 255), 10)
        if(min_x > i[0] - i[2]):
            min_x = x_p - r_p
        if(min_y > i[1] - i[2]):
            min_y = y_p - r_p
        if(max_x < i[0] + i[2]):
            max_x = x_p + r_p
        if(max_y < i[1] + i[2]):
            max_y = y_p + r_p
        # cv2.rectangle(img, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 5)
    print("圆心坐标", i[0], i[1])
    # 计算坐标
    gray = gray[min_y:max_y, min_x:max_x]

    cv2.imwrite(write_filename1, gray)
    cv2.imwrite(write_filename2, detected_edges)

if __name__ == "__main__":
    # 依次读取根目录下的每一个文件
    for file in os.listdir(root_dir):
        process(file)
    # process('0524.bmp')
