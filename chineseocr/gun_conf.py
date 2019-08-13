import logging
import logging.handlers
from logging.handlers import WatchedFileHandler
import os
import multiprocessing

workpath = os.path.dirname(__file__)
print(workpath)

bind = '0.0.0.0:5000'
# workers = multiprocessing.cpu_count() * 2 + 1
workers = 8
print('*'*50 + 'the number of workers is : {} worker'.format(workers), '*'*50)
backlog = 2048
worker_class = "gevent"
daemon = True
debug = True
reload = True
chdir = os.path.join(workpath, 'app')
proc_name = 'gunicorn.proc'

loglevel = 'debug'
access_log_format = '%(t)s %(p)s %(h)s "%(r)s" %(s)s %(L)s %(b)s %(f)s" "%(a)s"'
accesslog = os.path.join(workpath, "log/gunicorn_access.log")
errorlog = os.path.join(workpath, "log/gunicorn_error.log")
