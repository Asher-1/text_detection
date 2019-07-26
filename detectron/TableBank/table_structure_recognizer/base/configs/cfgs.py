# -*- coding: utf-8 -*-
import os

# ---------------------------------------- System_config
ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
print(20 * "++--")
print(ROOT_PATH)

MODEL_FILE = os.path.join(ROOT_PATH, "models/model.pt")
TABLE_PATH = os.path.join(ROOT_PATH, "tables/res.xlsx")
TEMP_PATH = os.path.join(ROOT_PATH, "tmp")
PRE_PATH = os.path.join(ROOT_PATH, "tmp/pred.txt")
IMAGE_INFO = os.path.join(ROOT_PATH, "tmp/info.txt")
LOG_PATH = os.path.join(ROOT_PATH + "log/log.log")

GPU_MODE = 0
VERBOSE = True
DATA_TYPE = "img"  # "img",  "text"

if __name__ == '__main__':
    print("RES_PATH = ", PRE_PATH)
    print("LOG_PATH = ", LOG_PATH)
    print("TEMP_PATH = ", TEMP_PATH)
    print("IMAGE_INFO = ", IMAGE_INFO)
    print("MODEL_FILE = ", MODEL_FILE)
    print("TABLE_PATH = ", TABLE_PATH)
