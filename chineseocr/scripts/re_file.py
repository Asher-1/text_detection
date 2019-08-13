#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chineseocr
FILE_NAME    :  re_file
AUTHOR       :  DAHAI LU
TIME         :  2019/7/26 上午9:04
PRODUCT_NAME :  PyCharm
"""

# import re
#
#
# def fuzzy_match(text, target, max_error_num=None):
#     import regex
#     if max_error_num is None:
#         max_error_num = len(target) // 2
#     res = regex.findall(r"(?:%s){e<=%s}" % (target, max_error_num), text)
#     if len(res) > 0:
#         return "".join(list(filter(lambda t: t in target, res[0])))
#     else:
#         return ""
#
#
# text = [" 经营范 计算机硬件领域内技术开发, 所 上海市松江区新宾镇新绿路398号 ",
#         "商务信息咨询，计算机软硬件及耗材批发零售",
#         "【依法须批准的项目】",
#         "登记机关"]
# N = len(text)
#
# res_dict = {}
# addString = []
# s_indx = 0
# for i in range(N):
#     txt = text[i].replace(' ', '')
#
#     if "登记机关" in txt:
#         if "经营范围" in res_dict.keys():
#             for j in range(s_indx, i):
#                 res_dict["经营范围"] += text[j].replace(' ', '')
#         break
#
#     t1 = fuzzy_match(txt, target="经营范围")
#     t2 = fuzzy_match(txt, target="住所")
#     if t1 == "":
#         continue
#     else:
#         pattern = r"%s(.*)%s" % (t1, t2)
#
#     res = re.findall(pattern, txt)
#
#     if len(res) > 0:
#         s_indx = i + 1
#         res_dict["经营范围"] = res[0].strip()
#
#     # 成立日期
#     res = re.findall(r"(?<=%s).+" % t2, txt)
#     if len(res) > 0:
#         key_word = ['住所', '省', '市', '县', '街', '村', '镇', '区', '城']
#         if any(key in key_word for key in res[0]):
#             addString.append(res[0].replace('住所', ''))
#         if len(addString) > 0:
#             res_dict['住所'] = ''.join(addString)
#
# print(res_dict)

import cv2
from apphelper.tools import his_equl_color1 as his_equl_color

path = "/home/yons/develop/AI/text_detection/chineseocr/test/idcard/IDCardOCR2.jpg"
img = cv2.imread(path)  # BGR
cv2.namedWindow("origin image", cv2.WINDOW_NORMAL)
cv2.imshow("origin image", img)
img = his_equl_color(img)
cv2.namedWindow("hist equalization", cv2.WINDOW_NORMAL)
cv2.imshow("hist equalization", img)

cv2.waitKey()
cv2.destroyAllWindows()
