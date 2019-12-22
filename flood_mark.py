import cv2
import numpy as np
from tkinter import filedialog
# mode有三种，0为进行泛洪运算，1为橡皮擦，2为仿射变换
mode = 0
color = (0, 0, 255)
lower = np.array([0, 253, 253])
upper = np.array([3, 255, 255])
hull = None
point = None
perspective_points = []  # 仿射变换标记的点
M = None  # 仿射变换使用的矩阵


# 回调函数
def draw_circle(event, x, y, flags, param):
    global mode, hull, point

    # 按下鼠标左键
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode == 0:
            cv2.floodFill(img_flood, None, (x, y), color, (scale, scale, scale), (scale, scale, scale), 4)
        elif mode == 1:
            img_flood[y-scale:y+scale, x-scale:x+scale] = img_background[y-scale:y+scale,x-scale:x+scale]
        elif mode == 2:
            perspective_points.append([x, y])
            if 2 <= len(perspective_points) <= 3:
                cv2.line(img_flood, tuple(perspective_points[len(perspective_points) - 2]), (x, y), (0,255,0), 1)
            elif len(perspective_points) == 4:
                cv2.line(img_flood, tuple(perspective_points[2]), tuple(perspective_points[3]), (0,255,0), 1)
                cv2.line(img_flood, tuple(perspective_points[3]), tuple(perspective_points[0]), (0,255,0), 1)
                perspective(*perspective_points)
    # 当鼠标左键按下并移动时
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if mode == 0:
            cv2.floodFill(img_flood, None, (x, y), color, (scale, scale, scale), (scale, scale, scale), 4)
        elif mode == 1:
            img_flood[y-scale:y+scale,x-scale:x+scale] = img_background[y-scale:y+scale, x-scale:x+scale]
    # 右键按下时进行凸包运算。
    elif event == cv2.EVENT_RBUTTONDOWN:
        bag()


def perspective(p1, p2, p3, p4):
    global M, img_flood, img_background, perspective_points
    pts1 = np.float32([p1, p2, p4, p3])  # 这里反向3、4，因为此处的点顺序为左上右上左下右下，标点是循序为顺时针
    pts2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    print(M)
    img_background = cv2.warpPerspective(img, M, (600, 600))
    img_flood = img_background
    perspective_points.clear()


def showSetting(simg):
    font = cv2.FONT_HERSHEY_SIMPLEX
    shape = simg.shape
    img_show = np.zeros((shape[0] + 100, shape[1], shape[2]), np.uint8)
    # 使用白色填充图片区域,默认为黑色
    img_show.fill(255)
    img_show[100:] = img_flood[:]
    cv2.putText(img_show, 'Press "m" to change the shape', (0, 30), font, 1, color, 2)
    cv2.rectangle(img_show, (0, 35), (1024, 68), (0, 0, 0), -1)  # 擦去上一个字符串，否则会出现重影
    cv2.putText(img_show, 'The shape you are drawing is :' + ('regtangle' if mode else 'circle'),
                (0, 60), font, 1, color, 2)
    cv2.putText(img_show, 'Click right button to change color', (0, 90), font, 1, color, 2)

    cv2.imshow('imagse', img_show)


def write():
    sfilename = save_filename()
    df = open(sfilename.split('.')[0] + '.csv', "w")
    # 接下来写入标记多边形的形心和各定点
    # df.write("center,vertexs" + "\n")
    df.write(str(point[0]) + "," + str(point[1]) + "\n")
    this_img = cv2.cvtColor(img_orgin, cv2.COLOR_GRAY2RGB)
    cv2.circle(this_img, point, 2, (0, 255, 255), -1)
    points = []
    for i in range(len(hull)):
        points.append(tuple(hull[i][0]))
        cv2.line(this_img, tuple(hull[i][0]), tuple(hull[(i + 1) % len(hull)][0]), (0, 255, 255), 2)
        df.write(str(tuple(hull[i][0])[0]) + "," + str(tuple(hull[i][0])[1]) + "\n")
    # 接下来写入标记多边形的周长和面积
    perimeter = 0
    for i in range(len(points)):
        perimeter += np.math.sqrt((points[i][0] - points[(i + 1) % len(points)][0]) ** 2 +
                                  (points[i][1] - points[(i + 1) % len(points)][1]) ** 2)
    area = poly_area(np.array(points).T[0], np.array(points).T[1])
    df.write("perimeter," + str(perimeter) + "\narea," + str(area))

    if sfilename.find('.') == -1:  # 如果没有写后缀
        cv2.imwrite(sfilename + '.bmp', this_img)
    else:
        cv2.imwrite(sfilename, this_img)


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def bag():
    """对img_flood上的红色区域进行凸包运算并标记"""
    global hull, point
    hsv = cv2.cvtColor(img_flood, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    # 寻找凸包并绘制凸包（轮廓）
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        length = len(hull)
        # 如果凸包点集中的点个数大于5
        if length > 5:
            # 绘制图像凸包的轮廓
            for i in range(length):
                cv2.line(img_flood, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 1)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            point = (cX, cY)
            cv2.circle(img_flood, (cX, cY), 2, (0, 255, 255), -1)


def open_filename():
    """返回文件名称，然后使用cv2的方法打开这个文件"""
    return filedialog.askopenfilename(title='打开单个文件',
                           filetypes=[("JPG图片", "*.jpg"), ('PNG图片', '*.png'), ('位图', '*.bmp')],  # 只处理的文件类型
                           initialdir='./')


def save_filename():
    """返回文件名称，然后使用cv2的方法保存这个文件"""
    return filedialog.asksaveasfilename(title='保存文件',
                            filetypes=[("JPG图片", "*.jpg"), ('PNG图片', '*.png'), ('位图', '*.bmp')],  # 只处理的文件类型
                            initialdir='./')


if __name__ == "__main__":
    ofilename = open_filename()
    scale = 1

    img = cv2.imread(ofilename, 0)
    img_orgin = img.copy()  # 保存一份原有的图像供reset功能以及擦除功能使用
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 这份图像会在之后被转换为彩色格式
    img_background = img.copy()
    img_flood = img.copy()  # 这份用来展示、进行泛洪,凸包运算
    cv2.namedWindow('image')
    linetype=cv2.LINE_AA
    cv2.setMouseCallback('image', draw_circle)

    while 1:
        cv2.imshow('image', img_flood)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('w'):
            # 将信息写入文件中
            write()
        elif k == ord('h'):
            img_background = cv2.equalizeHist(img_orgin)
            img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2RGB)
            img_flood = img_background.copy()
        elif k == ord('b'):
            img_background = cv2.blur(img_orgin, (5, 5), 0)
            img_background = cv2.cvtColor(img_background, cv2.COLOR_GRAY2RGB)
            img_flood = img_background.copy()
        elif k == ord('r'):
            img_background = cv2.cvtColor(img_orgin, cv2.COLOR_GRAY2RGB)
            img_flood = img_background.copy()
        elif k == ord('n'):
            ofilename = open_filename()
            img = cv2.imread(ofilename, 0)
            img_orgin = img.copy()  # 保存一份原有的图像供reset功能以及擦除功能使用
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 这份图像会在之后被转换为彩色格式
            img_background = img.copy()
            img_flood = img.copy()  # 这份用来展示、进行泛洪,凸包运算
        elif k == ord('1'):
            scale = 1
        elif k == ord('2'):
            scale = 2
        elif k == ord('3'):
            scale = 3
        elif k == ord('0'):
            scale = 0
        elif k == ord('m'):
            perspective_points.clear()  # 切换模式的时候，把仿射变换的点阵清空掉
            mode = (mode + 1) % 3
        elif k == 27:  # ESC
            break

    cv2.destroyAllWindows()
