#!/usr/bin/env bash
sudo docker load -i chineseocr_cpu.tar
sudo docker run -dit --name=ocr_cpu -p 8080:8080 -p 9999:9999 chineseocr-cpu:latest
