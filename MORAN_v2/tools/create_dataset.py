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
import lmdb  # install lmdb by "pip install lmdb"
import cv2
import numpy as np


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.fromstring(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k.encode(), v)


def createDataset(outputPath, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.

    ARGS:
        outputPath    : LMDB output path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    nSamples = len(labelList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        label_text = labelList[i]
        picture_and_label = label_text.strip().split(" ")
        imagePath = os.path.join(ROOT_PATH, picture_and_label[0])
        label = picture_and_label[1]
        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue
        with open(imagePath, 'rb') as f:
            imageBin = f.read()
        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
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


if __name__ == '__main__':
    # ROOT_PATH = "/media/yons/data/dataset/images/text_data/MORAN/raw_picture/ICDAR2003"
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/MORAN/raw_picture/ICDAR2013"
    # ROOT_PATH = "/media/yons/data/dataset/images/text_data/MORAN/raw_picture/ICDAR2015"
    outputPath = ROOT_PATH

    with open(os.path.join(ROOT_PATH, "gt_all.txt"), 'r') as f:
        labelList = f.readlines()
    createDataset(outputPath, labelList, lexiconList=None, checkValid=True)
    #	--train_nips "/media/yons/data/dataset/images/text_data/MORAN/raw_picture/ICDAR2013" \
    #	--train_cvpr "/media/yons/data/dataset/images/text_data/MORAN/raw_picture/ICDAR2015" \
    #	--valroot "/media/yons/data/dataset/images/text_data/MORAN/raw_picture/ICDAR2003" \