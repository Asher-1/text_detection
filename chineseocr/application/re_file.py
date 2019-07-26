#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chineseocr
FILE_NAME    :  re_file
AUTHOR       :  DAHAI LU
TIME         :  2019/7/26 上午9:04
PRODUCT_NAME :  PyCharm
"""

import re

text = [" 经营范围 计算机硬件领域内技术开发, 住所 上海市松江区新宾镇新绿路398号 ",
        "商务信息咨询，计算机软硬件及耗材批发零售",
        "【依法须批准的项目】",
        "登记机关"]
N = len(text)

res_dict = {}
addString = []
s_indx = 0
for i in range(N):
    txt = text[i].replace(' ', '')

    res = re.findall(r".*经营范围(.*?)住所", txt)
    if len(res) > 0:
        s_indx = i + 1
        res_dict["经营范围"] = res[0].strip()

    # 成立日期
    res = re.findall(r"(?<=住所).+", txt)
    if len(res) > 0:
        key_word = ['住所', '省', '市', '县', '街', '村', '镇', '区', '城']
        if any(key in key_word for key in res[0]):
            addString.append(res[0].replace('住所', ''))
        if len(addString) > 0:
            res_dict['住所'] = ''.join(addString)
    if "登记机关" in txt:
        if "经营范围" in res_dict.keys():
            for j in range(s_indx, i):
                res_dict["经营范围"] += text[j].replace(' ', '')
        break

print(res_dict)
