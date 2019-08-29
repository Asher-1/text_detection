#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chinese_ocr
FILE_NAME    :  merge_dict
AUTHOR       :  DAHAI LU
TIME         :  2019/8/23 下午1:42
PRODUCT_NAME :  PyCharm
"""

import os
from collections import Counter

if __name__ == '__main__':
    dict1 = "char_std_5072.txt"
    dict2 = "char_std_5990.txt"

    with open(dict1, "r") as fr1, open(dict2, "r") as fr2:
        voc1 = fr1.read().splitlines()
        voc2 = fr2.read().splitlines()

        c = Counter(voc2)
        print(c.most_common())

        print("origin length: {}".format(len(voc1)))
        voc_set = list(set(voc1))
        print("after length: {}".format(len(voc_set)))
        for char in voc1:
            if char not in voc2:
                print(char)
