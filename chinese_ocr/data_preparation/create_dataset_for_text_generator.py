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
import shutil
import datetime
from tqdm import tqdm


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


ROOT_PATH = "/media/yons/data/dataset/images/text_data/syn_chinese_data/"
CHAR_SET = "char_std_5072.txt"
IMAGE_PATH = ROOT_PATH + "text_images"
out_PATH = ROOT_PATH + "images2"
file_path = get_files_list(IMAGE_PATH)
image_indx = 0

TRAIN_TEST_RATIO = 0.8
train_num = int(len(file_path) * TRAIN_TEST_RATIO)
TRAIN_EXAMPLE = file_path[:train_num]
TEST_EXAMPLE = file_path[train_num:]

char_dict = {}
with open(CHAR_SET, "r") as f:
    char_set = f.read().splitlines()
    for ind, char in enumerate(char_set):
        if ind == 0:
            char_dict[" "] = ind
        else:
            char_dict[char] = ind


def generate_dataset(data, label_file):
    global image_indx
    ignore_num = 0
    with open(label_file, "w") as f:
        for file in tqdm(data):
            file_name = os.path.basename(file)
            file_ext = os.path.splitext(file_name)[-1]
            char_list = file_name.split("_")[0].split()
            char_num = len(char_list)
            if char_num < 2:
                ignore_num += 1
                continue
            if char_num > 15:
                print("invalid length : {}".format(char_num))
                continue

            random_string = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            file_name = "{}{}{}".format(random_string, image_indx, file_ext)
            image_indx += 1
            label = file_name + " " + " ".join([str(char_dict[char]) for char in char_list]) + "\n"
            shutil.copy(file, os.path.join(out_PATH, file_name))
            f.write(label)
    print("ignore number : {}".format(ignore_num))
    return image_indx


print("start generating training dataset")
generate_dataset(data=TRAIN_EXAMPLE, label_file=ROOT_PATH + "data_train2.txt")
print("start generating testing dataset")
generate_dataset(data=TEST_EXAMPLE, label_file=ROOT_PATH + "data_test2.txt")
