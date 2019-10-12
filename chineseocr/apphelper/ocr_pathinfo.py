#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chineseocr
FILE_NAME    :  pathinfo
AUTHOR       :  DAHAI LU
TIME         :  2019/8/1 上午9:12
PRODUCT_NAME :  PyCharm
"""
import os

# path configuration
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
TMP_FOLDER = os.path.join(ROOT_PATH, "tmp")
if not os.path.exists(TMP_FOLDER):
    os.makedirs(TMP_FOLDER)

# url configuration
OCR_WEB_URL = 'http://127.0.0.1:8080/ocr'
