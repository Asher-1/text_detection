#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  MORAN_v2
FILE_NAME    :  create_dataset
AUTHOR       :  DAHAI LU
TIME         :  2019/8/1 下午1:52
PRODUCT_NAME :  PyCharm
"""

import os
import sys
import glob
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np
from tqdm import tqdm


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    try:
        img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, file_list, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    nSamples = len(file_list)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for file_name, label in tqdm(list(file_list.items())):
        image_id = os.path.splitext(file_name)[0]
        image_path = os.path.join(IMAGE_DIR, file_name)

        try:
            if sys.version_info.major == 2:
                label = label.replace('\xef\xbb\bf', '')
                label = label.replace('\xe2\x80\x8d', '')
            else:
                label = label.replace('\ufeff', '')
                label = label.replace('\xef\xbb\xbf', '')
            label = label.strip()

            if VIS:
                img = cv2.imread(image_path)
                cv2.imwrite(os.path.join(VIS_DIR, "{}_{}".format(cnt, file_name)), img)

            with open(image_path, 'rb') as f:
                imageBin = f.read()
            if checkValid:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % image_path)
                    continue

            imageKey = 'image-%09d' % cnt
            labelKey = 'label-%09d' % cnt
            cache[imageKey] = imageBin
            cache[labelKey] = label.encode()
            if lexiconList:
                lexiconKey = 'lexicon-%09d' % cnt
                cache[lexiconKey] = ' '.join(lexiconList[i])
            if cnt % 1000 == 0:
                writeCache(env, cache)
                cache = {}
                print('Written %d / %d' % (cnt, nSamples))
            cnt += 1
        except Exception as e:
            import traceback
            traceback.print_exc()
            continue
    nSamples = cnt - 1
    cache['num-samples'] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def getFilePathList(file_dir):
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list


def get_files_list(file_dir, postfix='ALL'):
    postfix = postfix.split('.')[-1]
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix == 'ALL':
        file_list = filePath_list
    else:
        for file in filePath_list:
            basename = os.path.basename(file)
            postfix_name = basename.split('.')[-1]
            if postfix_name == postfix:
                file_list.append(file)
    file_list.sort()
    return file_list


def index_to_string(charset, texts):
    labels = ""
    for text in texts:
        labels += charset[int(text)]
    return labels


def readfile(filename):
    with open(map_path, 'r', encoding='utf-8') as f:
        char_set = f.read().splitlines()
    with open(filename, 'r') as f:
        res = f.read().splitlines()
    dic = {}
    for i in res:
        p = i.split(' ')
        char_label = index_to_string(char_set, p[1:])
        dic[p[0]] = char_label
    return dic


if __name__ == '__main__':
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/chinese_ocr_data"
    IMAGE_DIR = os.path.join(ROOT_PATH, "images")
    VIS_DIR = os.path.join(ROOT_PATH, "vis")
    # ANNOTATION_PATH = os.path.join(ROOT_PATH, "data_test.txt")
    ANNOTATION_PATH = os.path.join(ROOT_PATH, "data_train.txt")
    map_path = os.path.join(ROOT_PATH, "char_std_5990.txt")
    anno_file_list = readfile(ANNOTATION_PATH)

    VIS = False
    OUT_PATH = os.path.join(ROOT_PATH, "syn_train")
    createDataset(OUT_PATH, anno_file_list, lexiconList=None, checkValid=True)
