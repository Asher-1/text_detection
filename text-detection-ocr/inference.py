#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  text-detection-ocr
FILE_NAME    :  inference
AUTHOR       :  DAHAI LU
TIME         :  2019/8/13 上午10:09
PRODUCT_NAME :  PyCharm
"""

import os
import time
import dlocr


def inference_ctpn():
    from dlocr import ctpn
    ctpn = ctpn.get_or_create()
    ctpn.predict("asset/license/BizLicenseOCR2.jpg", "asset/demo_ctpn_license.jpg")


def inference_ocr():
    ocr = dlocr.get_or_create()
    # bboxes, texts = ocr.detect("asset/IDCardOCR1.jpg")
    bboxes, texts = ocr.detect("asset/license/BizLicenseOCR2.jpg")
    # bboxes, texts = ocr.detect("asset/license/license.png")
    print('\n'.join(texts))


def inference_densenet():
    from dlocr.densenet import load_dict, default_dict_path
    from dlocr import densenet
    densenet = densenet.get_or_create()
    text, img = densenet.predict("asset/1.jpg", load_dict(default_dict_path))
    print(text)


if __name__ == '__main__':
    start = time.time()
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
    inference_ctpn()
    inference_ocr()
    print(f"cost: {(time.time() - start) * 1000}ms")
