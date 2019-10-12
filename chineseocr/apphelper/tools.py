#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chineseocr
FILE_NAME    :  tools
AUTHOR       :  DAHAI LU
TIME         :  2019/7/29 下午1:22
PRODUCT_NAME :  PyCharm
"""
import os
import io
import cv2
import json
import regex
import errno
import shutil
import base64
import random
import datetime
import requests
import numpy as np
from PIL import Image


def fuzzy_match(text, target, max_error_num=None):
    if max_error_num is None:
        max_error_num = len(target) // 2
    res = regex.findall(r"(?:%s){e<=%s}" % (target, max_error_num), text)
    if len(res) > 0:
        return "".join(list(filter(lambda t: t in target, res[0])))
    else:
        return ""


def convert_pdf_to_image(pdf_file):
    import tempfile
    if isinstance(pdf_file, str):
        from pdf2image import convert_from_path
        with tempfile.TemporaryDirectory() as path:
            image_list = convert_from_path(pdf_file, output_folder=path)
    elif isinstance(pdf_file, bytes):
        from pdf2image import convert_from_bytes
        with tempfile.TemporaryDirectory() as path:
            image_list = convert_from_bytes(pdf_file, output_folder=path)
    else:
        raise TypeError("cannot parse unknown type: {}".format(type(pdf_file)))
    return image_list


def dir_nonempty(dirname):
    # If directory exists and nonempty (ignore hidden files), prompt for action
    return os.path.isdir(dirname) and len([x for x in os.listdir(dirname) if x[0] != '.'])


def mkdir_p(dirname):
    """ Like "mkdir -p", make a dir recursively, but do nothing if the dir exists
    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def empty_dir(dirname, parent=False):
    try:
        # shutil.rmtree(dirname, ignore_errors=True)
        if dir_nonempty(dirname):
            shutil.rmtree(dirname, ignore_errors=False)
    except Exception as e:
        print(e)
    finally:
        if not parent:
            mkdir_p(dirname)


# generate a unique id by datetime now
def generate_unique_id():
    now_str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    random_number = random.randint(0, 100)
    random_number_str = str(random_number)
    if random_number < 10:
        random_number_str = str(0) + str(random_number)
    now_random_str = now_str + "-" + random_number_str
    return now_random_str


def get_image_from_io(file):
    try:
        image_array = np.asarray(Image.open(io.BytesIO(file.read())))
        file.seek(0)
    except (OSError, NameError):
        return None
    return image_array


def preprocess_images(file, tmp_folder, ext_format):
    # from werkzeug import secure_filename
    # filename = secure_filename(file.filename)
    filename = file.filename
    file_extention = filename.split('.')[-1]

    file_path_list = []
    if file_extention == "pdf":
        pl_image_list = convert_pdf_to_image(file.read())
        # save_file_name = os.path.splitext(save_file_name)[0]
        for index, img in enumerate(pl_image_list):
            # tmp_name = os.path.join(tmp_folder, "{}_{}.{}".format(save_file_name, index, ext_format))
            ret, buf = cv2.imencode(".jpg", np.asarray(img))
            image_bytes = Image.fromarray(np.uint8(buf)).tobytes()
            file_path_list.append(image_bytes)
            # img.save(tmp_name, ext_format)
    elif file_extention in ["doc", "docx"]:
        from docx import Document
        save_file_name = "{}.{}".format(generate_unique_id(), file_extention)
        os.makedirs(tmp_folder)
        save_file_path = os.path.join(tmp_folder, save_file_name)
        file.save(save_file_path)
        doc = Document(save_file_path)
        for shape in doc.inline_shapes:
            content_id = shape._inline.graphic.graphicData.pic.blipFill.blip.embed
            content_type = doc.part.related_parts[content_id].content_type
            if not content_type.startswith("image"):
                continue
            img_data = doc.part.related_parts[content_id]._blob
            # save_file_name = os.path.basename(doc.part.related_parts[content_id].partname)
            # tmp_name = os.path.join(tmp_folder, save_file_name)
            file_path_list.append(img_data)
            # with open(tmp_name, 'wb') as fp:
            #     fp.write(img_data)
    else:
        file_path_list.append(file.read())
    return file_path_list


def read_img_base64(p):
    if isinstance(p, bytes):
        imgString = base64.b64encode(p)
    elif isinstance(p, str):
        with open(p, 'rb') as f:
            imgString = base64.b64encode(f.read())
    else:
        raise TypeError("cannot parse unknown type: {}".format(type(p)))
    imgString = b'data:image/jpeg;base64,' + imgString
    return imgString.decode()


def post_web(file, bill_model='通用OCR', url='http://127.0.0.1:8080/ocr', text_angle=False, text_line=False):
    res_dict = {}
    try:
        imgString = read_img_base64(file)
        param = {'billModel': bill_model,
                 'imgString': imgString,
                 'textAngle': text_angle,
                 'textLine': text_line, }
        param = json.dumps(param)
        req = requests.post(url, data=param)
        data = req.content.decode('utf-8')
        res_dict = json.loads(data)
    except Exception as e:
        print(e)
    finally:
        return res_dict


def his_equl_color1(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    channels = cv2.split(ycrcb)

    # https://docs.opencv.org/4.1.0/d5/daf/tutorial_py_histogram_equalization.html
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    clahe.apply(channels[0], channels[0])

    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img


def his_equl_color2(img):
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    cv2.equalizeHist(channels[0], channels[0])
    # 将处理后的通道和没有处理的两个通道合并，命名为ycrcb
    cv2.merge(channels, ycrcb)
    # 将YCrCb图像转换回RGB图像
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img
