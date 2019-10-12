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
    # URL = 'http://47.103.28.225:9999/image2text?'  ##url地址
    # URL = 'http://192.168.17.211:9999/image2text?'  ##url地址
    # URL = 'http://erp.erow.cn:22776/image2text?'  ##url地址

    files = {'file': open(p, 'rb')}
    URL += 'billModel={}&textAngle={}'.format(billModel, True)
    req = requests.post(URL, files=files)
    data = req.content.decode('utf-8')
    res_list = json.loads(data)
    for res_dict in res_list:
        print("#" * 50)
        time_take = res_dict['timeTake']
        state = res_dict['state']
        print("time take: {}s\tstate: {}".format(time_take, state))
        for res in res_dict["res"]:
            print("{}:\t{}".format(res["name"], res["text"]))


if __name__ == '__main__':
    # p = '../test/idcard/40039724617079296.jpg'
    # post(p, '身份证')

    # p = '../test/license/BizLicenseOCR2.jpg'
    # post(p, '通用OCR')

    # p = '../test/test_file/id5.docx'
    # post(p, '身份证')

    p = '../test/license/verifications.jpeg'
    post(p, '营业执照')

    # http://erp.erow.cn:22776/image2text?billModel={'身份证', '营业执照'}&textAngle=True
