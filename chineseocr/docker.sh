#!/usr/bin/env bash
##拉取基础镜像
docker build -t chineseocr .
##启动服务
docker run -d --runtime=nvidia --name=ocr -p 8080:8080 chineseocr /root/anaconda3/bin/python app.py


