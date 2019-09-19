#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  merge_dataset
AUTHOR       :  DAHAI LU
TIME         :  2019/8/14 下午4:59
PRODUCT_NAME :  PyCharm
"""


from shutil import copy as s_copy
from tqdm import tqdm
from os.path import join


def merge_dataset(out_file, input_1, input_2):
    with open(out_file, "w") as fw, open(input_1, "r") as fr1, open(input_2, "r") as fr2:
        data_list1 = fr1.read().splitlines()
        data_list2 = fr2.read().splitlines()
        for data in tqdm(data_list1):
            image_file = data.split()[0]
            image_path = join(IMAGE_PATH1, image_file)
            s_copy(image_path, OUT_PATH)
        for data in tqdm(data_list2):
            image_file = data.split()[0]
            image_path = join(IMAGE_PATH2, image_file)
            s_copy(image_path, OUT_PATH)

        merge_list = []
        merge_list.extend(data_list1)
        merge_list.extend(data_list2)
        for merge_line in merge_list:
            text = "{}\n".format(merge_line)
            fw.write(text)


if __name__ == '__main__':
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/syn_chinese_data/"

    IMAGE_PATH1 = ROOT_PATH + "images3"
    TRAIN_DATA1 = ROOT_PATH + "data_train3.txt"
    TEST_DATA1 = ROOT_PATH + "data_test3.txt"

    IMAGE_PATH2 = ROOT_PATH + "images4"
    TRAIN_DATA2 = ROOT_PATH + "data_train4.txt"
    TEST_DATA2 = ROOT_PATH + "data_test4.txt"

    OUT_PATH = ROOT_PATH + "syn_images2"
    TRAIN_OUT = ROOT_PATH + "syn_train2.txt"
    TEST_OUT = ROOT_PATH + "syn_test2.txt"

    print("start merging training dataset")
    merge_dataset(TRAIN_OUT, TRAIN_DATA1, TRAIN_DATA2)
    print("start merging testing dataset")
    merge_dataset(TEST_OUT, TEST_DATA1, TEST_DATA2)
