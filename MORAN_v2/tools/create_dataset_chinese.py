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
    for i in tqdm(range(nSamples)):
        anno_file = file_list[i]
        with open(anno_file, 'r', encoding='utf-8') as f:
            groundtruth = f.read().splitlines()
        image_id = os.path.splitext(os.path.basename(anno_file))[0]
        file_name = "{}.jpg".format(image_id)
        image_path = os.path.join(IMAGE_DIR, file_name)
        img = cv2.imread(image_path)

        for block in groundtruth:
            try:
                if sys.version_info.major == 2:
                    block = block.replace('\xef\xbb\bf', '')
                    block = block.replace('\xe2\x80\x8d', '')
                else:
                    block = block.replace('\ufeff', '')
                    block = block.replace('\xef\xbb\xbf', '')
                anno = block.strip().split(',')

                points = list(map(eval, anno[:8]))
                polygon = [[points[0], points[1]], [points[2], points[3]], [points[4], points[5]],
                           [points[6], points[7]]]
                x1, y1, w1, h1 = cv2.boundingRect(np.array(polygon))
                if anno[8] == "1":
                    continue

                label = anno[-1].split("\"")[1]
                temp_img = img[y1:y1 + h1, x1:x1 + w1]
                imagePath = "tmp.jpg"
                if VIS:
                    cv2.imwrite(os.path.join(VIS_DIR, "{}_{}".format(cnt, file_name)), temp_img)
                else:
                    cv2.imwrite(imagePath, temp_img)
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


if __name__ == '__main__':
    ROOT_PATH = "/media/yons/data/dataset/images/text_data/FOTS_TF"
    IMAGE_DIR = os.path.join(ROOT_PATH, "train")
    VIS_DIR = os.path.join(ROOT_PATH, "vis")
    ANNOTATION_PATH = os.path.join(ROOT_PATH, "train/*.txt")
    anno_file_list = list(glob.iglob(ANNOTATION_PATH))

    VIS = True
    OUT_PATH = os.path.join(ROOT_PATH, "output")
    createDataset(OUT_PATH, anno_file_list, lexiconList=None, checkValid=True)
