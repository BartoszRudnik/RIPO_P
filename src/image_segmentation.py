import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import pygame
import datetime as dt


class ImageSegmentation():
    def __init__(self):
        super().__init__()

    def car_position(self, frame, left_fit, right_fit):

        XM_PER_PIX = 3.5 / 480

        car_location = frame.shape[1] / 2
        height = frame.shape[0]

        bottom_left = left_fit[0]*height**2 + left_fit[
            1]*height + left_fit[2]

        bottom_right = right_fit[0]*height**2 + right_fit[
            1]*height + right_fit[2]

        center_lane = (bottom_right - bottom_left) / 2 + bottom_left

        center_offset = (np.abs(car_location) - np.abs(
            center_lane)) * XM_PER_PIX / 1000000

        self.center_offset = center_offset

        return center_offset

    def put_car_position_on_image(self, frame, car_position):

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_start_position = (10, 100)
        font_scale = 2
        font_color = (0, 0, 0)
        line_type = 2

        cv2.putText(frame, "Center offset: " + str(car_position)[:5] + " cm",
                    text_start_position, font, font_scale, font_color, line_type)

        return frame

    def put_lane_curvature_on_image(self, frame, left_curvature, right_curvature):

        font = cv2.FONT_HERSHEY_SIMPLEX
        text_start_position = (10, 250)
        font_scale = 2
        font_color = (0, 0, 0)
        line_type = 2

        cv2.putText(frame, "Curve radius: " + str((left_curvature + right_curvature) / 2)[:5] + " m",
                    text_start_position, font, font_scale, font_color, line_type)

        return frame

    def lane_curvature(self, left_y, left_x, right_y, right_x):
        y_eval = 1180
        YM_PER_PIX = 3.5 / 450
        XM_PER_PIX = 3.5 / 480

        left_fit_cr = np.polyfit(left_y * YM_PER_PIX, left_x * (
            XM_PER_PIX), 2)

        right_fit_cr = np.polyfit(right_y * YM_PER_PIX, right_x * (
            XM_PER_PIX), 2)

        left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*YM_PER_PIX + left_fit_cr[
                        1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curvem = ((1 + (2*right_fit_cr[
                        0]*y_eval*YM_PER_PIX + right_fit_cr[
            1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        return left_curvem, right_curvem

    def get_histogram(self, frame):

        histogram = np.sum(frame[int(frame.shape[0]/2):, :], axis=0)

        return histogram


    def get_histogram_peaks(self, histogram):

        histogram_mid = np.int(histogram.shape[0] / 2)

        left_peak = np.argmax(histogram[:histogram_mid])
        right_peak = np.argmax(histogram[histogram_mid:]) + histogram_mid

        return left_peak, right_peak


    def get_bird_view(self, frame, original_image_size):

        region_of_interest = np.array([[(250, 830), (0, 1280), (1080, 1280),  (730, 830)]])
        desired_points = np.array([[(250, 0), (250, 1280), (980, 1280), (980, 0)]])

        transformation_matrix = cv2.getPerspectiveTransform(region_of_interest, desired_points)
        inverse_transformation_matrix = cv2.getPerspectiveTransform(desired_points, region_of_interest)

        warped_frame = cv2.warpPerspective(frame, transformation_matrix, original_image_size, flags=(cv2.INTER_LINEAR))
        
        _, warped_frame_to_binary = cv2.threshold(warped_frame, 127, 255, cv2.THRESH_BINARY)

        return warped_frame_to_binary

    def canny(self, image):
        lane_image = np.copy(image)
        gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
        blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
        canny_image = cv2.Canny(blur_image, 80, 150)
        return canny_image

    def region_of_interest(self, image):
        trapezoid = np.array(
            [[(250, 830), (0, 1280), (1080, 1280),  (730, 830)]])
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, trapezoid, 255)
        masked_image = cv2.bitwise_and(image, mask)
        return masked_image

    def fill_lane(self, image, frame):
        points = np.array([[(image[0][2], image[0][3]), (image[0][0], image[0][1]),
                            (image[1][0], image[1][1]), (image[1][2], image[1][3])]])

        filled_image = np.copy(frame)
        cv2.fillPoly(filled_image, points, color=[0, 255, 0])

        return filled_image

    def display_lines(self, image, lines):
        check_lines = np.zeros_like(image)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.reshape(4)
                cv2.line(check_lines, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return check_lines

    def make_coordinates(self, image, line_params):
        try:
            slope, intercept = line_params

            if slope < 0.01 and slope > 0:
                slope = 1
            if slope > -0.01 and slope < 0:
                slope = -1

            y1 = 1280
            y2 = 890

            x1 = int((y1 - intercept)/slope)
            x2 = int((y2 - intercept)/slope)

            return True, np.array([x1, y1, x2, y2])

        except TypeError:
            return False, np.array([0, 0, 0, 0])

    def average_slope_intercept(self, image, lines):
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
        left_detected, left_line = self.make_coordinates(
            image, left_fit_average)

        right_fit_average = np.average(right_fit, axis=0)
        right_detected, right_line = self.make_coordinates(
            image, right_fit_average)

        car_position = self.car_position(image, left_line, right_line)

        left_line_y = np.array([left_line[1], left_line[3]])
        left_line_x = np.array([left_line[0], left_line[2]])

        right_line_y = np.array([right_line[1], right_line[3]])
        right_line_x = np.array([right_line[0], right_line[2]])

        if left_detected and right_detected:
            left_curvature, right_curvature = self.lane_curvature(
                left_line_y, left_line_x, right_line_y, right_line_x)
            return left_curvature, right_curvature, car_position, True, np.array([left_line, right_line])
        else:
            return 0, 0, car_position, False, np.array([left_line, right_line])

    def analize_video(self, frame):

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        canny_image = self.canny(frame)
        cropped_image = self.region_of_interest(canny_image)

        hough_image = cv2.HoughLinesP(
            cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        left_curvature, right_curvature, car_position, valid_detection, average_lines_image = self.average_slope_intercept(
            frame, hough_image)

        if not valid_detection:
            frame = self.put_car_position_on_image(frame, car_position)
            return False, frame

        filled_lanes = self.fill_lane(average_lines_image, frame)
        lines_image = self.display_lines(frame, average_lines_image)

        combo_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)
        combo_image = cv2.addWeighted(combo_image, 0.8, filled_lanes, 1, 1)
        combo_image = self.put_car_position_on_image(combo_image, car_position)
        combo_image = self.put_lane_curvature_on_image(
            combo_image, left_curvature, right_curvature)

        return True, combo_image

    def analize_photo(self, frame):

        canny_image = self.canny(frame)
        cropped_image = self.region_of_interest(canny_image)

        bird_view_image = self.get_bird_view(cropped_image, frame.shape[::-1][1:])

        hough_image = cv2.HoughLinesP(bird_view_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
        average_lines_image = self.average_slope_intercept(frame, hough_image)

        filled_lanes = self.fill_lane(average_lines_image, frame)
        lines_image = self.display_lines(frame, average_lines_image)

        combo_image = cv2.addWeighted(frame, 0.8, lines_image, 1, 1)
        combo_image = cv2.addWeighted(combo_image, 0.8, filled_lanes, 1, 1)

        return combo_image
