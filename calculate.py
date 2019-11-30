import re
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.special import comb, perm


def calculate_2():
    r_3s = []
    delta = 0.01
    for ploy in datas:
        n = 3
        c_x, c_y = ploy[0]
        r_3_sum = 0
        remain = 0  # 每段线剩下的长度
        dots = []
        for i in range(len(ploy[1])):
            x1, y1 = ploy[1][i]
            x2, y2 = ploy[1][(i + 1) % len(ploy[1])]
            if x1 == x2:
                y1, x1 = ploy[1][i]
                y2, x2 = ploy[1][(i + 1) % len(ploy[1])]
            k = (y2 - y1) / (x2 - x1)
            b = y2 - k * x2
            remain_x = remain / (k ** 2 + 1)
            delta_x = delta / (k ** 2 + 1)
            for j in range(round((np.math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)-remain) / delta - 0.5)):
                dot_x = x1 + remain_x + j * delta_x
                dot_y = k * dot_x + b
                dots.append([dot_x - c_x, dot_y - c_y])
            remain = np.math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) % 0.1
            # print(i, round(np.math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) / delta - 0.5))

        for i in range(len(dots) - 1):
            r_3 = (dots[i][0]**2 + dots[i][1]**2)**((n-1)/2)
            r_3 *= np.math.sqrt(((dots[i][0]-dots[i+1][0])**2 + (dots[i][1]-dots[i+1][1])**2))
            r_3_sum += r_3

        print(r_3_sum)

        r_3s.append(r_3_sum)


def calculate_4():
    r_3s = []
    m = 10000
    for ploy in datas:
        n = 4
        c_x, c_y = ploy[0]
        r_3_sum = 0
        for i in range(len(ploy[1])):
            x1, y1 = ploy[1][i]
            x2, y2 = ploy[1][(i + 1) % len(ploy[1])]
            l = np.math.sqrt((x2-x1)**2 + (y2-y1)**2)/m
            for j in range(m):
                dot_x = x1 + (x2 - x1) / m * (j+1/2) - c_x
                dot_y = y1 + (y2 - y1) / m * (j+1/2) - c_y
                r_3_sum += (dot_y**2 + dot_x**2)**((n-1)/2)*l
        print(str(r_3_sum) + '\t' + str((r_3_sum / np.math.pi / 2) ** (1/n)))
        r_3s.append(r_3_sum)


def calculate_5():
    j = 2
    r_5s = []
    for ploy in datas:
        r_5_sum = 0
        c_x, c_y = ploy[0]
        for i in range(len(ploy[1])):
            x1, y1 = ploy[1][i]
            x2, y2 = ploy[1][(i + 1) % len(ploy[1])]
            x1 -= c_x
            x2 -= c_x
            y1 -= c_y
            y2 -= c_y
            if x1 == x2:
                temp = x1
                x1 = y1
                y1 = temp
                temp = x2
                x2 = y2
                y2 = temp
            c = x1 - x2
            b = y1 - y2
            a = np.math.sqrt(b**2 + c**2) / abs(c)**(j*2+1)
            j_sum = 0
            for p in range(j+1):
                q_sum = 0
                for q in range(2*p+1):
                    q_value = comb(2*p, q)
                    q_value *= b**q * c**(2*j - 2*p)
                    q_value *= (c*y1 - b*x1)**(2*p - q)
                    q_value *= x2**(2*j - 2*p + q + 1) - x1**(2*j - 2*p + q + 1)
                    q_value /= 2*j - 2*p + q + 1
                    q_sum += q_value
                j_sum += comb(j, p) * q_sum
            r_5_sum += abs(a*j_sum)
        r_5s.append(r_5_sum)
        print(str(r_5_sum) + '\t' + str((r_5_sum / np.math.pi / 2) ** (1/(2*j+1))))


def calculate_3():
    r_3s = []
    for ploy in datas:
        r_3_sum = 0
        c_x, c_y = ploy[0]
        for i in range(len(ploy[1])):
            x1, y1 = ploy[1][i]
            x2, y2 = ploy[1][(i + 1) % len(ploy[1])]
            x1 -= c_x
            x2 -= c_x
            y1 -= c_y
            y2 -= c_y
            if x1 == x2:
                temp = x1; x1 = y1; y1 = temp
                temp = x2; x2 = y2; y2 = temp
            a = x1 - x2
            b = y1 - y2
            r3 = (a**2+b**2)*(x2**3-x1**3)/3
            r3 += 2*(b)*(x1*y2 - x2*y1)*(x2**2 - x1**2)/2
            r3 += (-a)*(x1*(b) + y1*(-a))**2
            r3 *= np.math.sqrt(b**2+a**2)/a**3
            r3 = abs(r3)
            r_3_sum += r3
        r_3s.append(r_3_sum)
        print(str(r_3_sum) + '\t' + str((r_3_sum / np.math.pi / 2) ** (1/3)))
        # print((r_3_sum / np.math.pi) ** (1/3))


def add_circle():
    p_num = 6
    delta_theta = 2 * np.math.pi / p_num
    pts = []
    for i in range(p_num):
        theta = i * delta_theta
        x = 1 * np.math.cos(theta)
        y = 1 * np.math.sin(theta)
        pts.append([x, y])
    datas.append([[0, 0], pts])


def calculate_1():
    perimeters = []  # 周长
    areas = []  # 面积
    for ploy in datas:
        perimeter = 0
        for i in range(len(ploy[1])):
            perimeter += np.math.sqrt((ploy[1][i][0] - ploy[1][(i + 1) % len(ploy[1])][0]) ** 2 +
                                      (ploy[1][i][1] - ploy[1][(i + 1) % len(ploy[1])][1]) ** 2)
        # perimeters.append(cv2.arcLength(ploy[1], True))
        area = PolyArea(np.array(ploy[1]).T[0], np.array(ploy[1]).T[1])
        perimeters.append(perimeter)
        areas.append(area)
        # print(perimeter, area)

        print(str(perimeter) + '\t' + str(area))

        res.append(perimeters)
        res.append(areas)


def draw():
    plt.plot(range(497, 608), res[0], linewidth=2.0)
    plt.show()
    plt.plot(range(497, 608), res[1], linewidth=2.0)
    plt.show()
    pass


def PolyArea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def read_file():
    for file in os.listdir(root_dir):
        if re.match('.*\.txt', file, flags=0):
            proportion = 8 / 1022 if int(file.split('.')[0]) <= 519 else 8 / 1034
            open_filename = root_dir + "\\" + file
            with open(open_filename, 'r') as fio:
                lines = fio.readlines()
                center = lines[0].split(':')[1]
                center = center.split(',')
                center = list((int(center[0][1:])*proportion, int(center[1][:-2])*proportion))
                # print(str(center[0]) + '\t' + str(center[1]))  # 坐标
                points = []
                for line in lines[1:]:
                    point = re.split('edge:*', line)[1]  # 有些忘记加分号所以使用正则表达式进行匹配
                    point = point.split(',')
                    point = list((int(point[0][1:])*proportion, int(point[1][:-2])*proportion))
                    points.append(point)
                datas.append([center, points])


def calculate_r():
    rjs = []
    for j in range(100):
        r_5s = []
        for ploy in datas:
            r_5_sum = 0
            c_x, c_y = ploy[0]
            for i in range(len(ploy[1])):
                x1, y1 = ploy[1][i]
                x2, y2 = ploy[1][(i + 1) % len(ploy[1])]
                x1 -= c_x
                x2 -= c_x
                y1 -= c_y
                y2 -= c_y
                if abs(x1 - x2) < 0.001:
                    temp = x1
                    x1 = y1
                    y1 = temp
                    temp = x2
                    x2 = y2
                    y2 = temp
                c = x1 - x2
                b = y1 - y2
                a = np.math.sqrt(b ** 2 + c ** 2) / abs(c) ** (j * 2 + 1)
                j_sum = 0
                for p in range(j + 1):
                    q_sum = 0
                    for q in range(2 * p + 1):
                        q_value = comb(2 * p, q)
                        q_value *= b ** q * c ** (2 * j - 2 * p)
                        q_value *= (c * y1 - b * x1) ** (2 * p - q)
                        q_value *= x2 ** (2 * j - 2 * p + q + 1) - x1 ** (2 * j - 2 * p + q + 1)
                        q_value /= 2 * j - 2 * p + q + 1
                        q_sum += q_value
                    j_sum += comb(j, p) * q_sum
                r_5_sum += abs(a * j_sum)
            r_5s.append((r_5_sum / np.math.pi / 2) ** (1 / (2 * j + 1)))
        rjs.append(r_5s)

    for i in range(len(rjs[0])):
        string = ''
        for j in range(len(rjs)):
            string += str(rjs[j][i]) + '\t'
        print(string)


if __name__ == '__main__':
    # root_dir为要读取文件的根目录
    root_dir = r".\data_2"
    datas = []  # n*2，n个点[i][0]为质心，[i][1]为一个多边形顶点的列表
    # datas.append([[4/3,1],[[0,3],[0,0],[4,0]]])
    # datas.append([[0,1/3],[[-1,0],[0,1],[1,0]]])
    # datas.append([[2/3,1/3],[[0,0],[1,1],[1,0]]])
    # datas.append([[1, 1],[[0,0],[2,0],[2,2],[0,2]]])
    # datas.append([[0, 0], [[-1, -1], [1, -1], [1, 1], [-1, 1]]])
    res = []  # 6*n，周长、面积、四个广义半径
    # read_file()
    add_circle()
    # calculate_1()  # 周长面积

    # calculate_2()  # 迭代法（不用了
    # calculate_3()  # 精确求广义半径
    # calculate_4()  # 迭代法
    # calculate_5()  # 精确求n=5
    calculate_r()

    # draw()
