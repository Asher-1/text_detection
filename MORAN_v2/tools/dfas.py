#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  MORAN_v2
FILE_NAME    :  dfas
AUTHOR       :  DAHAI LU
TIME         :  2019/8/12 下午1:27
PRODUCT_NAME :  PyCharm
"""
#
# import numpy as np
#
# data = [[2, 34, 6678, 7877], [23, 45, 6686, 6547], [1, 3, 4456675, 674576]]
#
# data_arr = np.array(data)
# x_min = np.min(data_arr[:, 0])
# y_min = np.min(data_arr[:, 1])
# x_max = np.max(data_arr[:, 2])
# y_max = np.max(data_arr[:, 3])
#
# print(x_min, y_min, x_max, y_max)


from config import charsets
from config import STOP_WORDS

text = ""
with open("char_std_5990.txt", "r") as f:
    char_std = f.read().splitlines()
    for char in char_std:
        if char not in STOP_WORDS:
            text += char

chinese_set = charsets + text

chinese_vob = "".join(list(set(chinese_set)))
print(len(chinese_vob))

with open("char_std.txt", "w") as f:
    f.write(chinese_vob)

