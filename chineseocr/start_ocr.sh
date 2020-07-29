#!/usr/bin/env bash
cd /chineseocr
/root/miniconda3/bin/gunicorn --chdir /chineseocr ocr_server:app -c /chineseocr/gun_ocr_conf.py &
/root/miniconda3/bin/python /chineseocr/app.py &
