#!/usr/bin/env bash
sudo docker stop ocr_gpu
sudo docker start ocr_gpu
sudo docker exec -d ocr_gpu /bin/bash /chineseocr/start_ocr.sh
