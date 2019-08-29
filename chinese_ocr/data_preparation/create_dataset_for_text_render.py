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
import numpy as np
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


def label_to_index(char_set):
    char_dict = {}
    with open(char_set, "r") as f:
        char_set = f.read().splitlines()
        for ind, char in enumerate(char_set):
            if ind == 0:
                char_dict[" "] = ind
            else:
                char_dict[char] = ind
    return char_dict


def generate_annotations(annotation, data, char_dict):
    with open(annotation, "w") as f:
        for anno in tqdm(data):
            image_id, char_string = anno.strip().split()
            file_name = "{}.{}".format(image_id, "jpg")
            # if len(char_string) < 2:
            #     print("the length of char is too short : {}".format(len(char_string)))
            #     continue
            if len(char_string) < 10:
                print("invalid length : {}".format(len(char_string)))
                continue
            label = file_name + " " + " ".join([str(char_dict[char]) for char in char_string]) + "\n"
            f.write(label)


def create_dataset():
    #  load label files
    with open(LABEL_TXT, "r") as f:
        file_list = f.read().splitlines()

    #  split dataset to train and test
    TRAIN_TEST_RATIO = 0.95
    file_index = list(range(len(file_list)))
    np.random.shuffle(file_index)

    train_num = int(len(file_index) * TRAIN_TEST_RATIO)
    file_array = np.array(file_list)
    TRAIN_EXAMPLE = file_array[file_index[:train_num]]
    TEST_EXAMPLE = file_array[file_index[train_num:]]

    print(
        "total samples: {}\n train samples: {}\n test_samples: {}".format(len(file_list), train_num, len(TEST_EXAMPLE)))

    #  create label to index dict
    label_index_map = label_to_index(CHAR_SET)

    #  generate train and test dataset
    generate_annotations(annotation=TRAIN_ANNOTATION_PATH, data=TRAIN_EXAMPLE, char_dict=label_index_map)
    generate_annotations(annotation=TEST_ANNOTATION_PATH, data=TEST_EXAMPLE, char_dict=label_index_map)


def create_keys():
    with open(CHAR_SET, "r") as fr, open("keys.txt", "w") as fw:
        char_set = fr.read().splitlines()
        char_string = "".join(char_set)
        fw.write(char_string)


if __name__ == '__main__':
    #  set path parameters
    # ROOT_PATH = "/media/yons/data/dataset/images/text_data/syn_chinese_data"
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data"
    CHAR_SET = "char_std_6266.txt"
    IMAGE_PATH = os.path.join(ROOT_PATH, "images2")
    LABEL_TXT = os.path.join(IMAGE_PATH, "tmp_labels.txt")

    TRAIN_ANNOTATION_PATH = os.path.join(ROOT_PATH, "data_train2.txt")
    TEST_ANNOTATION_PATH = os.path.join(ROOT_PATH, "data_test2.txt")

    create_dataset()
    # create_keys()
