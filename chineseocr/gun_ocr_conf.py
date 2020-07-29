import os
workpath = os.path.dirname(__file__)
print(workpath)

bind = '0.0.0.0:9999'
workers = 4
print('*'*50 + 'the number of workers is : {} worker'.format(workers), '*'*50)
backlog = 2048
worker_class = "gevent"
daemon = True
debug = True
reload = True
chdir = "/chineseocr"
proc_name = 'gunicorn.proc'

loglevel = 'debug'
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
accesslog = os.path.join(chdir, "log/ocr_access.log")
errorlog = os.path.join(chdir, "log/ocr_error.log")
