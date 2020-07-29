#!/usr/bin/env bash
##拉取基础镜像
docker build -t chineseocr-cpu .
##启动服务
#sudo docker run -d --name=ocr -p 8080:8080 -p 9999:9999 chineseocr-cpu /root/anaconda3/bin/python app.py
#sudo docker run -dit --name=ocr_cpu -p 8080:8080 -p 9999:9999 chineseocr-cpu:latest
sudo docker run -dit --name=ocr_cpu -p 8080:8080 -p 9999:9999 chineseocr-cpu:latest


