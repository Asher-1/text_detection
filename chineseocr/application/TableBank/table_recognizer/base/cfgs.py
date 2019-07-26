# -*- coding: utf-8 -*-
import os

# ---------------------------------------- System_config
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
WORK_PATH = os.path.join(ROOT_PATH, "table_recognizer/base")

MODEL_FILE = os.path.join(WORK_PATH, "models/model.pt")
LOG_PATH = os.path.join(WORK_PATH, "log/log.log")

TABLE_PATH = os.path.join(ROOT_PATH, "tables/res.xlsx")
TEMP_PATH = os.path.join(ROOT_PATH, "tmp")
PRE_PATH = os.path.join(ROOT_PATH, "tmp/pred.txt")
IMAGE_INFO = os.path.join(ROOT_PATH, "tmp/info.txt")

GPU_MODE = 0
VERBOSE = False
DATA_TYPE = "img"  # "img",  "text"

if __name__ == '__main__':
    print("ROOT_PATH = ", ROOT_PATH)
    print("WORK_PATH = ", WORK_PATH)
    print("RES_PATH = ", PRE_PATH)
    print("LOG_PATH = ", LOG_PATH)
    print("TEMP_PATH = ", TEMP_PATH)
    print("IMAGE_INFO = ", IMAGE_INFO)
    print("MODEL_FILE = ", MODEL_FILE)
    print("TABLE_PATH = ", TABLE_PATH)
