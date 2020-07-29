#!/usr/bin/env bash
sudo docker stop ocr_cpu
sudo docker start ocr_cpu
sudo docker exec -d ocr_cpu /bin/bash /chineseocr/start_ocr.sh
