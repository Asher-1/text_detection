#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  fiter
AUTHOR       :  DAHAI LU
TIME         :  2019/8/5 上午11:48
PRODUCT_NAME :  PyCharm
"""

# char_set = open('cn.txt', 'r', encoding='utf-8').readlines()
# char_set = list(set(char_set))
# with open("char_std_{}.txt".format(len(char_set)), "w") as f:
#     for char in char_set:
#         f.write(char)

import os
import cv2
import shutil


def getFilePathList(file_dir):
    '''
    获取file_dir目录下，所有文本路径，包括子目录文件
    :param rootDir:
    :return:
    '''
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir, postfix='ALL'):
    '''
    获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录
    :param file_dir:
    :param postfix:
    :return:
    '''
    postfix = postfix.split('.')[-1]
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix == 'ALL':
        file_list = filePath_list
    else:
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split('.')[-1]
            if postfix_name == postfix:
                file_list.append(file)
    file_list.sort()
    return file_list


ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr/"
IMAGE_PATH = ROOT_PATH + "out"
out_PATH = ROOT_PATH + "images"
file_path = get_files_list(IMAGE_PATH)
image_indx = 0

TRAIN_VAL_RATIO = 0.7
train_num = int(len(file_path)*TRAIN_VAL_RATIO)
TRAIN_EXAMPLE = file_path[:train_num]
VAL_EXAMPLE = file_path[train_num:]


char_dict = {}
with open("char_std_6283.txt", "r") as f:
    char_set = f.read().splitlines()[1:]
    for ind, char in enumerate(char_set):
        char_dict[char] = ind

with open(ROOT_PATH + "data_train.txt", "w") as f:
    for file in TRAIN_EXAMPLE:
        img = cv2.imread(file)
        img = cv2.resize(img, (518, 64))

        file_name = os.path.basename(file)
        file_ext = os.path.splitext(file_name)[-1]
        char_list = file_name.split("_")[0].split()
        file_name = "{}{}".format(image_indx, file_ext)
        image_indx += 1
        label = file_name + " " + " ".join([str(char_dict[char]) for char in char_list]) + "\n"
        cv2.imwrite(os.path.join(out_PATH, file_name), img)
        # shutil.copy(file, os.path.join(out_PATH, file_name))
        f.write(label)
with open(ROOT_PATH + "data_test.txt", "w") as f:
    for file in VAL_EXAMPLE:
        img = cv2.imread(file)
        img = cv2.resize(img, (518, 64))

        file_name = os.path.basename(file)

        file_ext = os.path.splitext(file_name)[-1]
        char_list = file_name.split("_")[0].split()
        file_name = "{}{}".format(image_indx, file_ext)
        image_indx += 1
        label = file_name + " " + " ".join([str(char_dict[char]) for char in char_list]) + "\n"
        cv2.imwrite(os.path.join(out_PATH, file_name), img)
        # shutil.copy(file, os.path.join(out_PATH, file_name))
        f.write(label)
