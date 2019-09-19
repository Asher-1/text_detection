#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  filter_datasets
AUTHOR       :  DAHAI LU
TIME         :  2019/8/20 下午5:19
PRODUCT_NAME :  PyCharm
"""

from os.path import join
from shutil import copy as s_copy
from tqdm import tqdm


def merge_dataset(out_file, input_file):
    with open(out_file, "w") as fw, open(input_file, "r") as fr:
        data_list = fr.read().splitlines()
        valid_number = 0
        for data in tqdm(data_list):
            line = data.split()
            lable = line[1:]
            if len(lable) != 15:
                continue
            image_file = line[0]
            image_path = join(IMAGE_PATH, image_file)
            s_copy(image_path, OUT_PATH)
            text = "{}\n".format(data)
            valid_number += 1
            fw.write(text)
        print("invalid number : {}".format(valid_number))


if __name__ == '__main__':
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/syn_chinese_data/"

    IMAGE_PATH = ROOT_PATH + "syn_images2"
    TRAIN_DATA = ROOT_PATH + "syn_train2.txt"
    TEST_DATA = ROOT_PATH + "syn_test2.txt"

    OUT_PATH = ROOT_PATH + "syn_images3"
    TRAIN_OUT = ROOT_PATH + "syn_train3.txt"
    TEST_OUT = ROOT_PATH + "syn_test3.txt"

    print("start merging training dataset")
    merge_dataset(TRAIN_OUT, TRAIN_DATA)
    print("start merging testing dataset")
    merge_dataset(TEST_OUT, TEST_DATA)
