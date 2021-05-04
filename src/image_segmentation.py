import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def canny(image):
    lane_image = np.copy(image)
    gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    canny_image = cv2.Canny(blur_image, 80, 150)
    return canny_image

def region_of_interest(image):   
    trapezoid = np.array([[(300, 830), (0, 1170), (1080, 1170),  (730, 830)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, trapezoid, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def fill_lane(image, frame):
    points = np.array([[(image[0][2], image[0][3]), (image[0][0], image[0][1]), (image[1][0], image[1][1]), (image[1][2], image[1][3])]])

    filled_image = np.copy(frame)
    cv2.fillPoly(filled_image, points, color=[0, 255, 0])

    return filled_image

def display_lines(image, lines):
    check_lines = np.zeros_like(image)
    if lines is not None:        
        for line in lines:
           x1, y1, x2, y2 = line.reshape(4)
           cv2.line(check_lines, (x1,y1), (x2,y2), (255, 0, 0), 10)        
    return check_lines

def make_coordinates(image, line_params):
    try:        
        slope, intercept = line_params

        if slope < 0.01 and slope > 0:
            slope = 1
        if slope > -0.01 and slope < 0:
            slope = -1

        y1 = 1170
        y2 = 890

        x1 = int((y1 - intercept)/slope)
        x2 = int((y2 - intercept)/slope)

        return np.array([x1, y1, x2, y2])   

    except TypeError:        
        slope, intercept = 0.001, 0         
        
        return np.array([0, 0, 0, 0])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            param = np.polyfit((x1, x2), (y1, y2), 1)

            slope = param[0]
            intercept = param[1]

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    
    left_fit_average = np.average(left_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
   
    right_fit_average = np.average(right_fit, axis=0)   
    right_line = make_coordinates(image, right_fit_average)

    return np.array([left_line, right_line])

cap = cv2.VideoCapture("nagranie.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)

    hough_image = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)    
    average_lines_image = average_slope_intercept(frame, hough_image)

    filled_lanes = fill_lane(average_lines_image, frame)
    lines_image = display_lines(frame, average_lines_image)

    combo_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)
    combo_image = cv2.addWeighted(combo_image, 0.8, filled_lanes, 1, 1)

    cv2.imshow("result", combo_image)
    cv2.waitKey(1)