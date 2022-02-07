from pdf2image import convert_from_path
import cv2
import numpy as np
import keyboard


def onMouse(event, x, y, flags, param):
    global count, click_points

    if event == cv2.EVENT_LBUTTONUP:
        click_points.append([y, x])
        count += 1

        if count == 2:
            keyboard.write('a', delay=0)


def findMultiple(before, now, standard):
    if now - before < radius:
        return 0
    else:
        mok = (now - before) // standard
        na = (now - before) % standard
        if na > radius:
            mok += 1
        return mok

path = "./print/정유/E-50-2B.pdf"
new_path = '.' + path.split('.')[1] + '.png'
convert_from_path(path)[0].save(new_path, 'PNG')

image = cv2.imread(new_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 파이프 부분만 오리기
count = 0
click_points = []
cv2.imshow('image0', image)
cv2.setMouseCallback('image0', onMouse)
cv2.waitKey()
cv2.destroyWindow('image0')

image = image[click_points[0][0]:click_points[1][0], click_points[0][1]:click_points[1][1]]
circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=20, minRadius=10, maxRadius=20)

li_for_col = []
li_for_row = []
circle_radius = dict()
circles = np.uint16(np.around(circles))

for i in circles[0, :]:
    cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
    li_for_col.append((i[0], i[1]))
    li_for_row.append((i[1], i[0]))
    if i[2] in circle_radius:
        circle_radius[i[2]] += 1
    else:
        circle_radius[i[2]] = 1


cv2.imshow('image', image)
key_pressed = cv2.waitKey()
if key_pressed == ord('q'):
    cv2.destroyAllWindows()
    exit(0)


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

for row in result:
    for elem in row:
        print(elem, end=' ')
    print()

cv2.waitKey()
cv2.destroyAllWindows()
