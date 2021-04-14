import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("zdjecie.png")

def canny(image):
    lane_image = np.copy(image)
    gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 80, 150)
    return canny_image

def region_of_interest(image):   
    trapezoid = np.array([[(363, 890), (20, 1170), (1007, 1170),  (675, 890)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, trapezoid, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def display_linees(image, lines):
    check_lines = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(check_lines, (x1,y1), (x2,y2), (255, 0, 0), 10)
    return check_lines

# canny_image = canny(image)
# cropped_image = region_of_interest(canny_image)

# lines_image = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# result_image = display_linees(image, lines_image)
# combo_image = cv2.addWeighted(image, 0.8, result_image, 1, 1)

# cv2.imwrite("cropped_image.png", cropped_image)

# cv2.imshow("result", combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture("nagranie.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)

    lines_image = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    result_image = display_linees(frame, lines_image)
    combo_image = cv2.addWeighted(frame, 0.8, result_image, 1, 1)

    cv2.imshow("result", combo_image)
    cv2.waitKey(1)