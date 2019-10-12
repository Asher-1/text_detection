#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
================================================================
Copyright (C) 2019 * Ltd. All rights reserved.
PROJECT      :  chineseocr
FILE_NAME    :  tdsfsd
AUTHOR       :  DAHAI LU
TIME         :  2019/10/12 上午11:21
PRODUCT_NAME :  PyCharm
================================================================
"""
import re
from apphelper.tools import fuzzy_match

txt = "生1995车5月13日"
res = re.findall(r'\d{4}(.?)\d{1,2}(.?)\d{1,2}(.?)', txt)
txt = txt.replace('出生', '')
for t in res[0]:
    txt = txt.replace(t, '-')
print(res)
print(txt)


print(txt[-4:])