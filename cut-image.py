from pdf2image import convert_from_path
import cv2
import numpy as np
import keyboard

path = "./print/Aro/Aro1/E-700-4A.pdf"
new_path = '.' + path.split('.')[1] + '.png'
convert_from_path(path)[0].save(new_path, 'PNG')

image = cv2.imread(new_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=10, maxRadius=20)
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)

cv2.imshow('image', image)
cv2.waitKey()
cv2.destroyAllWindows()