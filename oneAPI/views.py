import os
from django.http import JsonResponse, HttpResponse
from .findOne import findOne
from .models import equip as Equip
from .models import one as One
import base64


def oneFunction(one):
    one.convert_pdf_to_png()
    one.image_preprocessing()
    one.get_contour()

    one.detect_circle_size()
    one.cut_image_based_on_circle(one.contour)
    one.detect_circle(one.cut)

    one.check_diagonal()

    distinct_column = sorted(list(set(list(one.circles[0, :, 0]))))
    distinct_row = sorted(list(set(list(one.circles[0, :, 1]))))

    one.normal_column_list = one.coordinate_normalization(distinct_column, one.column_distance)
    one.normal_row_list = one.coordinate_normalization(distinct_row, one.row_distance)

    one.draw_pipe_array()
    one.make_json()

    one.remove_temp_files()


def oneApiView(request):
    if request.method == 'POST':
        one = findOne(request)
        oneFunction(one)

        return JsonResponse(one.json, status=200)

    else:
        return HttpResponse(status=405)


def listApiView(request):
    if request.method == 'GET':
        myjson = {}
        equip_list = []
        equips = Equip.objects.all()

        for eq in equips:
            equip_list.append(eq.equip_name)

        myjson['equip'] = equip_list

        return JsonResponse(myjson, status=200)

    elif request.method == 'POST':
        Equip.objects.all().delete()

        for walk in os.walk('./oneAPI/equips'):
            files = walk[2]
            for file in files:
                if file.split('.')[1] != 'pdf':
                    continue

                new_equip = Equip()
                equip_name = file.split('.')[0]
                new_equip.equip_name = equip_name
                new_equip.save()

        return HttpResponse(status=200)

    else:
        return HttpResponse(status=405)


def diffApiView(request):
    if request.method == 'GET':
        one = findOne(request)
        oneFunction(one)

        with open('./oneAPI/equips/' + one.equip + '.pdf', "rb") as img:
            image_data = base64.b64encode(img.read()).decode('utf-8')
        one.json['image'] = image_data

        return JsonResponse(one.json, status=200)

    elif request.method == 'POST':
        data = request.POST
        equip = data['equip']
        is_right = data['is_right']
        render_coordinates = data['render_coordinates']
        column_distance = data['column_distance']
        row_distance = data['row_distance']
        radius = data['radius']

        one = One.objects.filter(equip_name=equip)
        if len(one) != 0:
            one.delete()

        One(equip_name=equip, is_right=is_right, render_coordinates=render_coordinates, column_distance=column_distance, row_distance=row_distance, radius=radius).save()

        return HttpResponse(status=200)

    else:
        return HttpResponse(status=405)
