from pdf2image import convert_from_path
import cv2
import numpy as np
import os
from copy import deepcopy


def findMultiple(before, now, standard):
    if now - before < radius:
        return 0
    else:
        mok = (now - before) // standard
        na = (now - before) % standard
        if na > radius:
            mok += 1
        return mok


path = "print/Aro/Aro1/E-700-3A.pdf"
new_path = './' + path.split('.')[0] + '.png'
convert_from_path(path)[0].save(new_path, 'PNG')

original_image = cv2.imread(new_path)
image = original_image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur_image = cv2.medianBlur(gray, 7)

_, th = cv2.threshold(blur_image, 250, 255, cv2.THRESH_BINARY_INV)
# 숫자 크면 더 contour 넓게 잡힘
kernel = np.ones((18, 18), np.uint8)
dilation = cv2.dilate(th, kernel, iterations=2)
closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, 3)
hull = list()
for contour in contours:
    hull.append(cv2.convexHull(contour))

img_contour = cv2.drawContours(image, hull, -1, (255, 255, 255), -1)

img_contour = cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)
# param2 작을수록 개나소나 다 잡힘
circles = cv2.HoughCircles(img_contour, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=25, minRadius=8, maxRadius=20)

li_for_col = []
li_for_row = []
circle_radius = dict()
circles = np.uint16(np.around(circles))


most_l = 10000
most_r = 0
most_t = 10000
most_b = 0
for i in circles[0, :]:
    r = i[2]
    if i[0] - r < most_l:
        most_l = i[0] - r
    if i[0] + r > most_r:
        most_r = i[0] + r
    if i[1] - r < most_t:
        most_t = i[1] - r
    if i[1] + r > most_b:
        most_b = i[1] + r
    li_for_col.append((i[0], i[1]))
    li_for_row.append((i[1], i[0]))
    if r in circle_radius:
        circle_radius[r] += 1
    else:
        circle_radius[r] = 1
print(most_l, most_r, most_t, most_b)

cv2.imshow('image', img_contour)
cv2.waitKey()


radius = max(circle_radius, key=circle_radius.get)
li_for_col.sort()
li_for_row.sort()

difference_col = dict()
for ind, (i, j) in enumerate(li_for_col):
    if ind != (len(li_for_col) - 1):
        diff = li_for_col[ind+1][0] - i
        if diff == 0 or diff < radius:
            continue
        if diff in difference_col:
            difference_col[diff] += 1
        else:
            difference_col[diff] = 1

standard_col = max(difference_col, key=difference_col.get)

difference_row = dict()
for ind, (j, i) in enumerate(li_for_row):
    if ind != (len(li_for_row) - 1):
        diff = li_for_row[ind+1][0] - j
        if diff == 0 or diff < radius:
            continue
        if diff in difference_row:
            difference_row[diff] += 1
        else:
            difference_row[diff] = 1

standard_row = max(difference_row, key=difference_row.get)

i_set = set()
j_set = set()
for i, j in li_for_col:
    i_set.add(i)
    j_set.add(j)
i_distinct = sorted(list(i_set))
j_distinct = sorted(list(j_set))


i_indexing = []
for ind in range(len(i_distinct)):
    if ind == 0:
        i_indexing.append([i_distinct[ind]])
        continue
    value = findMultiple(i_distinct[ind-1], i_distinct[ind], standard_col)
    if value == 0:
        i_indexing[len(i_indexing)-1].append(i_distinct[ind])
    elif value == 1:
        i_indexing.append([i_distinct[ind]])
    else:
        for _ in range(value-1):
            i_indexing.append([])
        i_indexing.append([i_distinct[ind]])

j_indexing = []
for ind in range(len(j_distinct)):
    if ind == 0:
        j_indexing.append([j_distinct[ind]])
        continue
    value = findMultiple(j_distinct[ind-1], j_distinct[ind], standard_row)
    if value == 0:
        j_indexing[len(j_indexing)-1].append(j_distinct[ind])
    elif value == 1:
        j_indexing.append([j_distinct[ind]])
    else:
        for _ in range(value-1):
            j_indexing.append([])
        j_indexing.append([j_distinct[ind]])


result = [[0 for col in range(len(i_indexing))] for row in range(len(j_indexing))]
for i, j in li_for_col:
    for ind, temp_li in enumerate(i_indexing):
        if i in temp_li:
            i_ind = ind
    for ind, temp_li in enumerate(j_indexing):
        if j in temp_li:
            j_ind = ind
    result[j_ind][i_ind] = 1


col = 1
client_list = deepcopy(result)
for col_index in range(len(client_list)):
    row = 1
    for row_index in range(len(client_list[0])):
        if client_list[col_index][row_index] == 1:
            client_list[col_index][row_index] = (col, row)
            row += 1
    col += 1

# for row in client_list:
#     for elem in row:
#         print(elem, end=' ')
#     print()
# print()

for row in result:
    for elem in row:
        print(elem, end=' ')
    print()


if(os.path.isfile(new_path)):
    os.remove(new_path)

cv2.destroyAllWindows()
