#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  dlfd
AUTHOR       :  DAHAI LU
TIME         :  2019/8/14 下午4:45
PRODUCT_NAME :  PyCharm
"""

import os
import shutil
import random
from tqdm import tqdm

# ROOT_PATH = "/media/yons/data/dataset/images/text_data/syn_chinese_data/"
ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data/"
image_path = ROOT_PATH + "syn_images2"
out_file = ROOT_PATH + "syn_test2.txt"
temp_path = ROOT_PATH + "tmp"
LABEL_TXT = os.path.join(image_path, "tmp_labels.txt")
LABEL_OUT = os.path.join(ROOT_PATH, "tmp_labels.txt")

# with open(out_file, "w") as fw, open(ROOT_PATH + "data_test1.txt", "r") as fr1, \
#         open(ROOT_PATH + "data_test2.txt", "r") as fr2, \
#         open(ROOT_PATH + "syn_test0.txt", "r") as fr3:
#     data_list1 = fr1.read().splitlines()
#     data_list2 = fr2.read().splitlines()
#     data_list3 = fr3.read().splitlines()
#     merge_list = []
#     merge_list.extend(data_list1)
#     merge_list.extend(data_list2)
#     merge_list.extend(data_list3)
#
#     random.shuffle(merge_list)
#     for merge_line in tqdm(merge_list):
#         text = "{}\n".format(merge_line)
#         fw.write(text)

image_path = "/media/yons/data/dataset/images/text_data/chinese_ocr_data/syn_images"

# file_list = os.listdir(image_path)
# for file in file_list:
#     os.remove(os.path.join(image_path, file))

#
# with open(ROOT_PATH + "data_train2.txt", "r") as fr, open(ROOT_PATH + "train.txt", "w") as fw:
#     file_list = fr.read().splitlines()
#     invalid_num = 0
#     for line in tqdm(file_list):
#         line_list = line.split()
#         char_list = line_list[1:]
#         char_num = len(char_list)
#         if char_num < 2:
#             invalid_num += 1
#             file = line_list[0]
#             file_path = os.path.join(image_path, file)
#             shutil.move(file_path, temp_path)
#         else:
#             text = "{}\n".format(line)
#             fw.write(text)
#     print("invalid number : {}".format(invalid_num))
#
# file_list = os.listdir(temp_path)
#
# for file in tqdm(file_list):
#     file_name = os.path.join(temp_path, file)
#     new_file = os.path.join(temp_path, "{}_{}".format("tmp", file))
#     os.rename(file_name, new_file)
# print(file_name)
# os.remove(file_name)

# with open(LABEL_TXT, "r") as f:
#     file_list = f.read().splitlines()
#
# with open(LABEL_OUT, "w") as f:
#     for anno in tqdm(file_list):
#         image_id, char_string = anno.strip().split()
#         new_image_id = "{}_{}".format("0904_nums", image_id)
#         file_name = "{}.{}".format(image_id, "jpg")
#         new_file_name = "{}.{}".format(new_image_id, "jpg")
#
#         text = "{}\n".format(anno.replace(image_id, new_image_id))
#
#         old_file_path = os.path.join(image_path, file_name)
#         new_file_path = os.path.join(image_path, new_file_name)
#
#         os.rename(old_file_path, new_file_path)
#         f.write(text)


# ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data/"
# image_path = ROOT_PATH + "syn_images2"
# LABEL_TXT = os.path.join(ROOT_PATH, "syn_train2.txt")
# LABEL_OUT = os.path.join(ROOT_PATH, "syn_train3.txt")
# with open(LABEL_TXT, "r") as f:
#     file_list = f.read().splitlines()
#
# duplicated_num = 0
# with open(LABEL_OUT, "w") as f:
#     for anno in tqdm(file_list):
#         file_name = anno.strip().split()[0]
#         image_id = os.path.splitext(file_name)[0]
#         old_file_path = os.path.join(image_path, file_name)
#         if int(image_id) <= 200000:
#             try:
#                 os.remove(old_file_path)
#             except Exception as e:
#                 duplicated_num += 1
#                 print("delete file : {} again...".format(old_file_path))
#                 continue
#         else:
#             text = "{}\n".format(anno)
#             f.write(text)
