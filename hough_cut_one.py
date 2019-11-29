import cv2
import numpy as np
import os

# root_dir为要读取文件的根目录
root_dir = r".\imgs"
min_x, min_y = 9999, 9999
max_x, max_y = 0, 0
x, y, r = 0, 0, 0
i1 = []

def process(filename):
    global min_x, min_y, max_x, max_y, x, y, r, i1
    open_filename = root_dir + "\\" + filename
    write_filename1 = ".\\hough_light_res_512" + "\\" + filename
    write_filename2 = ".\\hough_light_circle_512" + "\\" + filename
    img = cv2.imread(open_filename, 0)
    gray = img[20:1100, 420:1650]
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
                                500, param1=88, param2=20, minRadius=480, maxRadius=520)
    circles = circles1[0, :, :]
    circles = np.uint16(np.around(circles))

    for i in circles[:]:
        cv2.circle(detected_edges, (i[0], i[1]), i[2], (255, 0, 0), 5)
        cv2.circle(detected_edges, (i[0], i[1]), 2, (255, 0, 255), 10)
        i1 = i
        print(i)
        x = i[0];  y = i[1]; r = i[2]
        if(min_x > i[0] - i[2]):
            min_x = i[0] - i[2]
        if(min_y > i[1] - i[2]):
            min_y = i[1] - i[2]
        if(max_x < i[0] + i[2]):
            max_x = i[0] + i[2]
        if(max_y < i[1] + i[2]):
            max_y = i[1] + i[2]
        # cv2.rectangle(img, (i[0] - i[2], i[1] + i[2]), (i[0] + i[2], i[1] - i[2]), (255, 255, 0), 5)
    print("圆心坐标", i[0], i[1])
    # 计算坐标
    gray = gray[min_y:max_y, min_x:max_x]

    cv2.imwrite(write_filename1, gray)
    cv2.imwrite(write_filename2, detected_edges)

def cut(filename):
    open_filename = root_dir + "\\" + filename
    write_filename1 = ".\\hough_light_res_512" + "\\" + filename
    write_filename2 = ".\\hough_light_circle_512" + "\\" + filename
    img = cv2.imread(open_filename, 0)
    gray = img[20:1100, 420:1650]
    # cv2.imwrite(write_filename2 + '_real.bmp', gray)
    gray = cv2.resize(gray, (0, 0), fx=0.905, fy=1)
    circle = gray.copy()
    cv2.circle(circle, (i1[0], i1[1]), i1[2], (255, 0, 0), 5)
    cv2.circle(circle, (i1[0], i1[1]), 2, (255, 0, 255), 10)
    cv2.imwrite(write_filename2, circle)
    gray = gray[min_y:max_y, min_x:max_x]


    cv2.imwrite(write_filename1, gray)



if __name__ == "__main__":
    # 依次读取根目录下的每一个文件
    process('0512.bmp')
    for file in os.listdir(root_dir):
        cut(file)

