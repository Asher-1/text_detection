#!/usr/bin/env bash
gunicorn -c gun_ocr_conf.py ocr_server:app &
python app.py &
