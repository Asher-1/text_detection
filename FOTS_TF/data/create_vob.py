#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  FOTS_TF
FILE_NAME    :  create_vob
AUTHOR       :  DAHAI LU
TIME         :  2019/8/6 上午11:18
PRODUCT_NAME :  PyCharm
"""

import os
import sys
import glob
import numpy as np
from tqdm import tqdm
from vocabulary import firth_vob
from vocabulary import secondth_vob
from vocabulary import tradition_vob


def load_annoataion(p):
    '''
    load annotation from the text file
    :param p:
    :return:
    '''
    # print p
    text_polys = []
    text_tags = []
    labels = []
    with open(p, 'r') as f:
        for line in f.readlines():
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            # line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            if sys.version_info.major == 2:
                line = line.replace('\xef\xbb\bf', '')
                line = line.replace('\xe2\x80\x8d', '')
            else:
                line = line.replace('\ufeff', '')
                line = line.replace('\xef\xbb\xbf', '')
            line = line.strip()
            line = line.split(',')
            if len(line) == 0:
                continue
            try:
                temp_line = map(eval, line[:8])
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, temp_line)
            except Exception as e:
                import traceback
                traceback.print_exc()
                print("float parse error : ", list(temp_line))
                print("{}: {}".format(p, line))
                continue
            difficulty = line[8]
            label = line[-1].split("\"")[1]
            labels.append(label)
        return labels


def create_vocabulary():
    char_dict = {}
    with open("char_std_6283.txt", "r", encoding='utf-8') as fr, open("char_std.txt", "w") as fw:
        char_list = fr.read().splitlines()
        str_label = ""
        txt_file_list = glob.iglob(TEXT_FILE)
        for file in tqdm(list(txt_file_list)):
            labels = "".join(load_annoataion(file))
            for label in labels:
                if label not in char_list:
                    char_list.append(label)

        for char in tqdm(char_list):
            if char in stop_words:
                continue
            str_label += char
        fw.write(str_label)
        print("the length of vob is : {}".format(len(str_label)))


def create_dicts():
    char_dict = {}
    with open("char_std.txt", "w") as fw:
        filter_labels = ""
        char_label = set()
        txt_file_list = glob.iglob(TEXT_FILE)
        for file in tqdm(list(txt_file_list)):
            labels = "".join(load_annoataion(file))
            for label in labels:
                if label in stop_words:
                    continue
                if label not in chinese_vob:
                    filter_labels += label
                    continue
                char_label.add(label)

        char_list = list(char_label)
        fw.write("".join(char_list))
        print("the length of vob is : {}".format(len(char_list)))
        print("".join(list(set(filter_labels))))
        print("the length of filter_labels is : {}".format(len(list(set(filter_labels)))))


if __name__ == '__main__':
    TEXT_FILE = "/media/yons/data/dataset/images/text_data/FOTS_TF/train/*.txt"
    stop_words = u"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ _-~～`'.·:;^/／|\"\'√!！?？$%％#@Ⅱ￥℃&*()∣<>——【】×《》（）[]{}▼_￣+=,。°，：；、”"

    chinese_vob = firth_vob + secondth_vob + tradition_vob
    print(len(chinese_vob))
    chinese_vob = "".join(list(set(chinese_vob)))
    print(len(chinese_vob))
    # create_vocabulary()

    create_dicts()
