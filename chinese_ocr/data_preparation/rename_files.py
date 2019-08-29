#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  rename_files
AUTHOR       :  DAHAI LU
TIME         :  2019/8/15 下午6:16
PRODUCT_NAME :  PyCharm
"""

import os
import shutil
from tqdm import tqdm

ROOT_PATH = "/media/yons/data/dataset/images/text_data/syn_chinese_data/"
image_path = ROOT_PATH + "images"
temp_path = ROOT_PATH + "tmp"
LABEL_TXT = os.path.join(image_path, "tmp_labels.txt")
LABEL_OUT = os.path.join(ROOT_PATH, "tmp_labels.txt")

if __name__ == '__main__':

    with open(LABEL_TXT, "r") as f:
        file_list = f.read().splitlines()

    with open(LABEL_OUT, "w") as f:
        for anno in tqdm(file_list):
            image_id, char_string = anno.strip().split()
            new_image_id = "{}_{}".format("0821_chn", image_id)
            file_name = "{}.{}".format(image_id, "jpg")
            new_file_name = "{}.{}".format(new_image_id, "jpg")

            text = "{}\n".format(anno.replace(image_id, new_image_id))

            old_file_path = os.path.join(image_path, file_name)
            new_file_path = os.path.join(image_path, new_file_name)

            os.rename(old_file_path, new_file_path)
            f.write(text)
