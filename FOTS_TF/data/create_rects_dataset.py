#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  FOTS_TF
FILE_NAME    :  create_dataset
AUTHOR       :  DAHAI LU
TIME         :  2019/8/6 下午2:53
PRODUCT_NAME :  PyCharm
"""
import os
import cv2
import json
import shutil
import glob
import numpy as np
from tqdm import tqdm


def get_roi_img(img, ploygon):
    rect = cv2.minAreaRect(ploygon)
    box = np.int0(cv2.boxPoints(rect))
    Xs = [i[0] for i in box]
    Ys = [i[1] for i in box]
    x1 = min(Xs)
    x2 = max(Xs)
    y1 = min(Ys)
    y2 = max(Ys)
    hight = y2 - y1
    width = x2 - x1
    cropImg = np.copy(img[y1:y1 + hight, x1:x1 + width])
    return cropImg


def verify_coordinates(polygon, image_h, image_w):
    left_top, right_top, right_bottom, left_bottom = polygon
    new_left_top = [round(max(0, left_top[0])), round(max(0, left_top[1]))]
    new_right_top = [round(min(image_w - 1, right_top[0])), round(max(0, right_top[1]))]
    new_right_bottom = [round(min(image_w - 1, right_bottom[0])), round(min(image_h - 1, right_bottom[1]))]
    new_left_bottom = [round(max(0, left_bottom[0])), round(min(image_h - 1, left_bottom[1]))]
    return [new_left_top, new_right_top, new_right_bottom, new_left_bottom]


def polygon_area(poly):
    '''
    compute area of a polygon
    :param poly:
    :return:
    '''
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(polygon):
    '''
    check so that the text poly is in the same direction,
    and also filter some invalid polygons
    :param polys:
    :param tags:
    :return:
    '''
    global invalid_poly, wrong_direction

    polys = np.array(polygon)
    if polys.shape[0] == 0:
        print("poly is empty...")
        return None
    p_area = polygon_area(polys)
    if abs(p_area) < 1:
        # print poly
        print('invalid poly\n', " p_area: {}".format(p_area))
        invalid_poly += 1
        return None
    if p_area > 0:
        print('poly in wrong direction')
        wrong_direction += 1
        polys = polys[(0, 3, 2, 1), :]
        if polygon_area(polys) > 0:
            print('poly in wrong direction again...')
            return None
    return polys


def create_datasets():
    global skipped, count
    for json_file in tqdm(list(json_file_list)):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                groundtruth = f.read()
            anno = json.loads(groundtruth.strip())
            image_id = os.path.splitext(os.path.basename(json_file))[0]
            label_file = os.path.join(OUT_PATH, "{}.txt".format(image_id))
            file_name = "{}.jpg".format(image_id)
            image_path = os.path.join(IMAGE_DIR, file_name)
            if VIS:
                img = cv2.imread(image_path)
                image_h, image_w = img.shape[:2]

            # load image jpeg data
            empty = True
            with open(label_file, "w", encoding="utf-8") as f:
                for block in anno['lines']:
                    points = block["points"]
                    polygon = [[points[0], points[1]], [points[2], points[3]], [points[4], points[5]],
                               [points[6], points[7]]]
                    poly_array = check_and_validate_polys(polygon)
                    if poly_array is None:
                        location = np.array(polygon).flatten()
                        img = cv2.imread(image_path)
                        cv2.polylines(img, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 1)
                        cv2.imwrite(os.path.join(VIS_PATH, file_name), img)
                        continue
                    location = poly_array.flatten()
                    if VIS:
                        cv2.polylines(img, [np.int0(location).reshape((-1, 1, 2))], True, (0, 255, 0), 1)

                    label = block['transcription']
                    empty = False
                    # write annotations
                    text = ""
                    for x, y in poly_array.tolist():
                        if VIS:
                            assert (0 <= x < image_w) and (0 <= y < image_h)
                        else:
                            assert 0 <= x and y >= 0
                        text += "{},{},".format(int(x), int(y))
                    difficulty = str(block['ignore'])
                    f.write("{}{},\"{}\"\n".format(text, difficulty, label))
                    f.flush()
            if empty:
                print("find empty images")
                os.remove(label_file)
            else:
                shutil.copy(image_path, OUT_PATH)
                if VIS:
                    cv2.imwrite(os.path.join(VIS_PATH, file_name), img)
                count += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            skipped += 1
            continue


if __name__ == '__main__':
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/ICDAR2019/RECTS/ReCTS"
    JSON_PATH = os.path.join(ROOT_PATH, "gt/*.json")
    json_file_list = glob.iglob(JSON_PATH)

    OUT_PATH = os.path.join(ROOT_PATH, "output")
    VIS = False
    VIS_PATH = os.path.join(ROOT_PATH, "vis")
    start_index = 0
    num_images = -1

    # load groundtruth file
    IMAGE_DIR = os.path.join(ROOT_PATH, "img")

    count = 0
    skipped = 0
    wrong_direction = 0
    invalid_poly = 0

    create_datasets()

    print("total processed samples : {}".format(count))
    print("skipped samples : {}".format(skipped))
    print("wrong_direction samples : {}".format(wrong_direction))
    print("invalid_poly samples : {}".format(invalid_poly))
