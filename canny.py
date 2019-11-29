import cv2
import numpy as np

def CannyThreshold(lowThreshold):
    gray = img
    detected_edges = cv2.GaussianBlur(gray,(25,25),0)
    detected_edges = cv2.equalizeHist(detected_edges)
    detected_edges = cv2.Canny(detected_edges,lowThreshold,lowThreshold*ratio,apertureSize = kernel_size)
    dst = cv2.bitwise_and(gray,gray,mask = detected_edges)  # just add some colours to edges from original image.
    cv2.imshow('canny demo',dst)

lowThreshold = 0
max_lowThreshold = 400
ratio = 3
kernel_size = 5

img = cv2.imread(r'.\all_res\0505.bmp', 0)

cv2.namedWindow('canny demo')

cv2.createTrackbar('Min threshold','canny demo',lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()