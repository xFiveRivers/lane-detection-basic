import cv2
import numpy as np
import matplotlib.pyplot as plt

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def roi(img):
    height = img.shape[0]
    traingle = np.array([])

img = cv2.imread('example_image.jpg')
img_arr = np.copy(img)
canny_img = canny(img_arr)

plt.imshow(canny_img)
plt.show()