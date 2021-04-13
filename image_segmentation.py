import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("zdjecie.png")

lane_image = np.copy(image)
gray_image = cv2.cvtColor(lane_image, cv2.COLOR_RGB2GRAY)
blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
canny_image = cv2.Canny(blur_image, 50, 150)

plt.imshow(canny_image)
plt.show()