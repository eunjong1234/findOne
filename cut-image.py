from pdf2image import convert_from_path
import cv2
import numpy as np


path = "./print/Aro/Aro1/E-700-4A.pdf"
new_path = '.' + path.split('.')[1] + '.png'
convert_from_path(path)[0].save(new_path, 'PNG')

original_image = cv2.imread(new_path)
image = original_image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.medianBlur(gray, 7)

_, th = cv2.threshold(blur_image, 250, 255, cv2.THRESH_BINARY_INV)
kernel = np.ones((15, 15), np.uint8)
dilation = cv2.dilate(th, kernel, iterations=2)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, 3)
hull = list()
for contour in contours:
    hull.append(cv2.convexHull(contour))
img_contour = cv2.drawContours(image, hull, -1, (255, 255, 255), -1)


img_contour = cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(img_contour, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=10, maxRadius=20)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv2.circle(img_contour, (i[0], i[1]), 2, (0, 0, 255), 3)


cv2.imshow('원본', original_image)
cv2.imshow('contour', img_contour)

cv2.waitKey()
cv2.destroyAllWindows()