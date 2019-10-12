#!/usr/bin/env bash
gunicorn -k gevent -c gun_ocr_conf.py ocr_server:app