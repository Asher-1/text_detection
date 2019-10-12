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


def post(p, billModel='通用OCR'):
    URL = 'http://127.0.0.1:9999/image2text?'  ##url地址
    files = {'file': open(p, 'rb')}
    URL += 'billModel={}&textAngle={}'.format(billModel, True)
    req = requests.post(URL, files=files)
    data = req.content.decode('utf-8')
    res_list = json.loads(data)
    for res_dict in res_list:
        print("#" * 50)
        for res in res_dict["res"]:
            print("{}:\t{}".format(res["name"], res["text"]))


if __name__ == '__main__':
    # p = 'test/idcard/IDCardOCR2.jpg'
    # post(p, '身份证')

    # p = '../test/license/BizLicenseOCR2.jpg'
    # post(p, '通用OCR')

    # p = '../test/license/verifications.jpeg'
    # post(p, '营业执照')

    p = '../test/test_file/yingye.pdf'
    post(p, '营业执照')
