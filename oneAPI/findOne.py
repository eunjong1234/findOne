from pdf2image import convert_from_path
import cv2
import numpy as np
import os
from django.core.files.storage import default_storage
from django.core import serializers
import json


class findOne:
    def __init__(self, request):
        self.json = {}
        self.is_diagonal = None
        self.renderCoordinates = None
        self.path = None
        self.equip = None

        self.origin = None
        self.gray = None
        self.blur = None
        self.contour = None
        self.cut = None

        self.radius = None
        self.circles = None

        self.column_distance = None
        self.row_distance = None
        self.normal_column_list = None
        self.normal_row_list = None
        self.result_array = None

        if len(request.FILES) == 1:
            self.file = request.FILES['pdf']
            self.equip = str(self.file.name).split('.')[0]
            self.path = './oneAPI/media/oneAPI/' + self.equip
            default_storage.save(self.path + '.pdf', self.file)
        else:
            self.equip = request.GET['equip']
            self.path = './oneAPI/equips/' + self.equip

    def convert_pdf_to_png(self):
        pages = convert_from_path(self.path + '.pdf')
        for page in pages:
            page.save(self.path + '.png', 'PNG')

    def remove_temp_files(self):
        for file in os.listdir('./oneAPI/media/oneAPI/'):
            os.remove('./oneAPI/media/oneAPI/' + file)

    def image_preprocessing(self):
        self.origin = cv2.imread(self.path + '.png')
        gray = cv2.cvtColor(self.origin, cv2.COLOR_BGR2GRAY)
        self.blur = cv2.medianBlur(gray, 7)

    def get_contour(self, contour_size=18):
        _, th = cv2.threshold(self.blur, 250, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((contour_size, contour_size), np.uint8)
        dilation = cv2.dilate(th, kernel, iterations=2)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

        contours, hierarchy = cv2.findContours(closing, cv2.RETR_EXTERNAL, 3)
        hull = list()
        for contour in contours:
            hull.append(cv2.convexHull(contour))

        img_contour = cv2.drawContours(self.origin, hull, -1, (255, 255, 255), -1)
        self.contour = cv2.cvtColor(img_contour, cv2.COLOR_BGR2GRAY)

    def detect_circle_size(self):
        pixel_max = 0
        radius = 0
        for rad in range(5, 90, 1):
            circles = cv2.HoughCircles(self.contour, cv2.HOUGH_GRADIENT, 1, rad * 2 - 1, param1=300, param2=15,
                                       minRadius=rad, maxRadius=rad)
            if circles is None:
                continue
            circles = np.uint16(np.around(circles))
            temp = self.contour.copy()
            for i in circles[0]:
                cv2.circle(temp, (i[0], i[1]), i[2], (255, 255, 255), 3)

            if temp.sum() > pixel_max:
                pixel_max = temp.sum()
                radius = rad
                self.circles = circles

        self.radius = radius

    def detect_circle(self, image):
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 2 * self.radius - 1, param1=300, param2=10,
                                   minRadius=self.radius, maxRadius=self.radius)
        self.circles = np.uint16(np.around(circles))

    def cut_image_based_on_circle(self, image):
        most_l = 10000
        most_r = 0
        most_t = 10000
        most_b = 0
        second_most_b = 0
        most_b_count = 1
        for i in self.circles[0, :]:
            r = i[2]
            if i[0] - r < most_l:
                most_l = i[0] - r
            if i[0] + r > most_r:
                most_r = i[0] + r
            if i[1] - r < most_t:
                most_t = i[1] - r
            if i[1] + r > most_b:
                if most_b > second_most_b:
                    second_most_b = most_b
                most_b = i[1] + r
                most_b_count = 1
            elif i[1] + r == most_b:
                most_b_count += 1
            if most_b > i[1] + r > second_most_b:
                second_most_b = i[1] + r

        if ((most_b - second_most_b) // (self.radius * 2) > 3) and (most_b_count == 1):
            most_b = second_most_b

        self.cut = image[most_t - 10:most_b + 10, most_l - 10:most_r + 10]

    def get_radius(self):
        radius_dict = dict()
        for radius in self.circles[0, :, 2]:
            if radius in radius_dict:
                radius_dict[radius] += 1
            else:
                radius_dict[radius] = 1

        self.radius = max(radius_dict, key=radius_dict.get)

    def get_distance_center(self, coordinate) -> int:
        difference_dict = dict()
        for index in range(len(coordinate)):
            if index != (len(coordinate) - 1):
                diff = coordinate[index + 1] - coordinate[index]
                if diff == 0 or diff < (self.radius / 2):
                    continue
                if diff in difference_dict:
                    difference_dict[diff] += 1
                else:
                    difference_dict[diff] = 1

        mymin = min(difference_dict, key=difference_dict.get)
        pop_list = []
        for key in difference_dict.keys():
            if key > (mymin * 1.5):
                pop_list.append(key)
        for key in pop_list:
            difference_dict.pop(key)
        return max(difference_dict, key=difference_dict.get)

    def find_multiple(self, before, now, standard):
        if (now - before) < (standard / 2):
            return 0
        else:
            mok = (now - before) // standard
            na = (now - before) % standard
            if na >= (standard * 0.7):
                mok += 1
            return mok

    def coordinate_normalization(self, distinct_coordinates, standard):
        indexing = []
        for ind in range(len(distinct_coordinates)):
            if ind == 0:
                indexing.append([distinct_coordinates[ind]])
                continue
            value = self.find_multiple(distinct_coordinates[ind - 1], distinct_coordinates[ind], standard)
            if value == 0:
                indexing[len(indexing) - 1].append(distinct_coordinates[ind])
            elif value == 1:
                indexing.append([distinct_coordinates[ind]])
            else:
                for _ in range(value - 1):
                    indexing.append([])
                indexing.append([distinct_coordinates[ind]])
        return indexing

    def draw_pipe_array(self):
        self.result_array = [[0 for _ in range(len(self.normal_column_list))] for _ in range(len(self.normal_row_list))]
        for i, j in self.circles[0, :, 0:2]:
            for ind, temp_li in enumerate(self.normal_column_list):
                if i in temp_li:
                    i_ind = ind
            for ind, temp_li in enumerate(self.normal_row_list):
                if j in temp_li:
                    j_ind = ind
            self.result_array[j_ind][i_ind] = 1

        self.renderCoordinates = [[0 for _ in range(len(self.normal_column_list))] for _ in range(len(self.normal_row_list))]
        row_num = 1
        for row_index in range(len(self.renderCoordinates)):
            col_num = 1
            is_change = False
            for col_index in range(len(self.renderCoordinates[0])):
                if self.result_array[row_index][col_index] == 1:
                    self.renderCoordinates[row_index][col_index] = [row_num, col_num]
                    col_num += 1
                    if not is_change:
                        is_change = True
            if is_change:
                row_num += 1

    def check_diagonal(self):
        self.is_diagonal = False

        self.column_distance = self.get_distance_center(sorted(list(self.circles[0][:, 0])))
        self.row_distance = self.get_distance_center(sorted(list(self.circles[0][:, 1])))

        if int(self.column_distance) < (self.radius * 1.9) or int(self.row_distance) < (self.radius * 1.9):
            self.is_diagonal = True

    def make_json(self):
        self.json['renderCoordinates'] = json.dumps(self.renderCoordinates)
        self.json['columnDistance'] = json.dumps(int(self.column_distance))
        self.json['rowDistance'] = json.dumps(int(self.row_distance))
        self.json['radius'] = json.dumps(int(self.radius))
