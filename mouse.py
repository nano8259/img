import cv2
import numpy as np
# 当鼠标按下时变为True
mode = True
color = (0, 0, 255)
lower = np.array([0, 253, 253])
upper = np.array([3, 255, 255])
hull = None
point = None


# 回调函数
def draw_circle(event, x, y, flags, param):
    global mode, hull, point

    # 按下鼠标左键时返回起始位置坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        if mode:
            cv2.floodFill(img, None, (x, y), color, (scale, scale, scale), (scale, scale, scale), 4)
        else:
            cv2.circle(img, (x, y), 4, (255,255,255), -1)
    # 当鼠标左键按下并移动时绘制图形。event可以查看移动，flag查看是否按下
    elif event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON:
        if mode:
            cv2.floodFill(img, None, (x, y), color, (scale, scale, scale), (scale, scale, scale), 4)
        else:
            cv2.circle(img, (x, y), 4, (255, 255, 255), -1)
    # 右键按下时进行凸包运算。
    elif event == cv2.EVENT_RBUTTONDOWN:
        bag()


def showSetting(img):
    # font = cv2.FONT_HERSHEY_SIMPLEX
    # cv2.putText(img, 'Press "m" to change the shape', (0, 30), font, 1, color, 2)
    # cv2.rectangle(img, (0, 35), (1024, 68), (0, 0, 0), -1) # 擦去上一个字符串，否则会出现重影
    # cv2.putText(img, 'The shape you are drawing is :' + ('regtangle' if mode else 'circle'),
    #             (0, 60), font, 1, color, 2)
    # cv2.putText(img, 'Click right button to change color', (0, 90), font, 1, color, 2)
    pass

def write(fileno):
    df = open(".\\data_2\\" + fileno + ".txt", "w")
    df.write("center:" + str(point) + "\n")
    this_img = orgin_img.copy()
    for i in range(len(hull)):
        cv2.line(this_img, tuple(hull[i][0]), tuple(hull[(i + 1) % len(hull)][0]), (0, 255, 255), 2)
        cv2.circle(this_img, point, 2, (0, 255, 255), -1)
        df.write("edge:" + str(tuple(hull[i][0])) + "\n")

    cv2.imwrite(".\\data_2\\" + fileno + ".bmp", this_img)


def bag():
    print('yes')
    global hull, point
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    # cv2.imshow('msk', mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    # 寻找凸包并绘制凸包（轮廓）
    for cnt in contours:
        hull = cv2.convexHull(cnt)
        length = len(hull)
        # 如果凸包点集中的点个数大于5
        if length > 5:
            # 绘制图像凸包的轮廓
            for i in range(length):
                cv2.line(img, tuple(hull[i][0]), tuple(hull[(i + 1) % length][0]), (0, 255, 0), 1)
            M = cv2.moments(cnt)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            point = (cX, cY)
            cv2.circle(img, (cX, cY), 2, (0, 255, 255), -1)


if __name__ == "__main__":
    fileno = '0575'
    scale = 1

    img = cv2.imread('.\\all_res\\' + fileno + '.bmp', 0)
    img = cv2.equalizeHist(img)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    orgin_img = img.copy()
    # img = cv2.blur(img, (5,5), 0)
    cv2.namedWindow('image')
    linetype=cv2.LINE_AA
    showSetting(img)
    cv2.setMouseCallback('image', draw_circle)
    while 1:
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('w'):
            write(fileno)
        elif k == ord('1'):  # ESC
            scale = 1
        elif k == ord('2'):  # ESC
            scale = 2
        elif k == ord('3'):  # ESC
            scale = 3
        elif k == ord('4'):  # ESC
            scale = 0
        elif k == ord('m'):
            mode = False if mode else True
        elif k == ord('m'):
            bag()
        elif k == 27:  # ESC
            break
    cv2.destroyAllWindows()
