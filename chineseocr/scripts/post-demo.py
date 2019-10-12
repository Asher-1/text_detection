# -*- coding: utf-8 -*-
"""
@author: asher
后台通过接口调用服务，获取OCR识别结果
"""
import base64
import requests
import json


def read_img_base64(p):
    with open(p, 'rb') as f:
        imgString = base64.b64encode(f.read())
    imgString = b'data:image/jpeg;base64,' + imgString
    return imgString.decode()


def post_ocr(p, billModel='通用OCR'):
    URL = 'http://127.0.0.1:8080/ocr'  ##url地址
    # URL = 'http://127.0.0.1:5000/image2text'  ##url地址
    imgString = read_img_base64(p)
    headers = {}
    param = {'billModel': billModel,  ##目前支持三种 通用OCR/ 火车票/ 身份证/
             'imgString': imgString, }
    param = json.dumps(param)
    req = requests.post(URL, data=param, headers=None)
    data = req.content.decode('utf-8')
    res_dict = json.loads(data)
    for res in res_dict["res"]:
        print("{}:\t{}".format(res["name"], res["text"]))


def post_img2text(p, billModel='通用OCR'):
    URL = 'http://127.0.0.1:9999/image2text'  ##url地址
    file = open(p, 'rb')
    files = {'file': file}
    param = {'billModel': billModel,  ##目前支持三种 通用OCR/ 火车票/ 身份证/
             'textAngle': True, }
    req = requests.post(URL, params=param, files=files)
    data = req.content.decode('utf-8')
    res_dict = json.loads(data)
    result = res_dict[0]
    time_take = result['timeTake']
    print("time take: {}".format(time_take))
    for res in result["res"]:
        print("{}:\t{}".format(res["name"], res["text"]))


if __name__ == '__main__':
    p = 'test/idcard/IDCardOCR2.jpg'
    post_ocr(p, '身份证')

    p = 'test/license/BizLicenseOCR2.jpg'
    post_ocr(p, '营业执照')

    # p = 'test/test_file/df.pdf'
    # post_img2text(p, '通用OCR')
