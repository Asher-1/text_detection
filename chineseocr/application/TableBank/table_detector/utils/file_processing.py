#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
PROJECT      :  chineseocr
FILE_NAME    :  file_processing
AUTHOR       :  DAHAI LU
TIME         :  2019/7/25 下午4:47
PRODUCT_NAME :  PyCharm
"""
import os
import errno
import shutil


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


def empty_dir(dirname):
    try:
        # shutil.rmtree(dirname, ignore_errors=True)
        if dir_nonempty(dirname):
            shutil.rmtree(dirname, ignore_errors=False)
    except Exception as e:
        print(e)
    finally:
        mkdir_p(dirname)
