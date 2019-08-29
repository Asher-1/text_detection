#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  merge_dataset
AUTHOR       :  DAHAI LU
TIME         :  2019/8/14 下午4:59
PRODUCT_NAME :  PyCharm
"""

import os
import time
import queue
import shutil
import random
from tqdm import tqdm
from multi_threading_util import MultiThreadHandler


def task(data, result_queue):
    dst_path = data[0]
    image_file = data[1].split()[0]
    image_path = os.path.join(dst_path, image_file)
    shutil.copy(image_path, OUT_PATH)
    result_queue.put(image_file)


def multi_generator(files_lists, dst_path):
    for file in files_lists:
        task_queue.put((dst_path, file))
    MultiThreadHandler(task_queue, task, out_queue, thread_num).run(True)
    # while True:
    #     print('Image : {} augmentation completely successfully...'.format(out_queue.get()))
    #     if out_queue.empty():
    #         break


def merge_dataset(out_file, input_1, input_2):
    with open(out_file, "w") as fw, open(input_1, "r") as fr1, open(input_2, "r") as fr2:
        data_list1 = fr1.read().splitlines()
        data_list2 = fr2.read().splitlines()

        merge_list = []
        merge_list.extend(data_list1)
        merge_list.extend(data_list2)

        random.shuffle(merge_list)
        for merge_line in tqdm(merge_list):
            text = "{}\n".format(merge_line)
            fw.write(text)

        multi_generator(data_list1, IMAGE_PATH1)
        multi_generator(data_list2, IMAGE_PATH2)


if __name__ == '__main__':
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data/"

    IMAGE_PATH1 = ROOT_PATH + "images"
    TRAIN_DATA1 = ROOT_PATH + "data_train.txt"
    TEST_DATA1 = ROOT_PATH + "data_test.txt"

    IMAGE_PATH2 = ROOT_PATH + "images2"
    TRAIN_DATA2 = ROOT_PATH + "data_train2.txt"
    TEST_DATA2 = ROOT_PATH + "data_test2.txt"

    OUT_PATH = ROOT_PATH + "syn_images2"
    TRAIN_OUT = ROOT_PATH + "syn_train.txt"
    TEST_OUT = ROOT_PATH + "syn_test.txt"

    thread_num = 64
    task_queue = queue.Queue()
    out_queue = queue.Queue()

    print("start merging training dataset")
    start = time.time()
    merge_dataset(TRAIN_OUT, TRAIN_DATA1, TRAIN_DATA2)
    print("merging training dataset time span: {} s".format(time.time()-start))

    print("start merging testing dataset")
    start = time.time()
    merge_dataset(TEST_OUT, TEST_DATA1, TEST_DATA2)
    print("merging testing dataset time span: {} s".format(time.time()-start))
