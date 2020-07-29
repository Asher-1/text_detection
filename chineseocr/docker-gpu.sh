#!/usr/bin/env bash
##拉取基础镜像
docker build -t chineseocr-gpu .
##启动服务
#sudo docker run -d --runtime=nvidia --name=ocr -p 8080:8080 chineseocr-gpu /root/anaconda3/bin/python app.py
sudo docker run -dit --runtime=nvidia --name=ocr_gpu -p 8080:8080 -p 9999:9999 chineseocr-gpu:latest
#sudo docker run -dit --runtime=nvidia --name=ocr_gpu -p 8080:8080 -p 9999:9999 --mount type=bind,source=/media/yons/data/tmp,target=/chineseocr/tmp chineseocr-gpu:latest


