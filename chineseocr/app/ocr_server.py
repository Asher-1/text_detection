# -*- coding: UTF-8 -*-

debug_mode = False
if not debug_mode:
    from gevent import monkey
    from gevent.pywsgi import WSGIServer
    monkey.patch_all()

import os
import sys
import logging
import traceback
from apphelper import tools
from projectinfo import ocr_pathinfo

from flask_restful import Api
from flask_restful import Resource
from flask_restful import reqparse
from flask import Flask, request, g


work_dir = os.path.dirname(os.path.dirname(__file__))

sys.path.append(work_dir)

# the flask app
app = Flask(__name__)
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# app.config.update(DEBUG=True)
app.config['TMP_FOLDER'] = ocr_pathinfo.TMP_FOLDER
if not os.path.exists(ocr_pathinfo.TMP_FOLDER):
    os.makedirs(ocr_pathinfo.TMP_FOLDER)
app.config['OCR_WEB_URL'] = ocr_pathinfo.OCR_WEB_URL

# the settings for uploading
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'doc', 'docx'}


# define the allowed file for uploading
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


# restful api
api = Api(app)


######*******************************************************************************************######
######*******************************************************************************************######
######*******************************************************************************************######
# default root url path
class Hello(Resource):
    def get(self):
        return {"message": "hello ai server"}


######*******************************************************************************************######
######*******************************************************************************************######
######*******************************************************************************************######
class Image2Text(Resource):
    # my_face_ai = FaceAI(FACE_CONFIG_FILE_PATH)
    # my_face_ai = FaceAI(face_utils_base_info.FACE_CONFIG_FILE_PATH)
    def __init__(self):
        self.parser = reqparse.RequestParser()
        self.parser.add_argument('billModel', type=str, location='args', required=False, default='通用OCR')
        self.parser.add_argument('textAngle', type=bool, location='args', required=False, default=False)
        self.parser.add_argument('textLine', type=bool, location='args', required=False, default=False)

    def post(self):
        url_params = self.parser.parse_args()
        bill_model = url_params.get('billModel')
        text_angle = url_params.get('textAngle')
        text_line = url_params.get('textLine')
        tmp_folder = app.config['TMP_FOLDER']
        web_url = app.config['OCR_WEB_URL']

        file = request.files['file']
        if not allowed_file(file.filename):
            return {"message": "file extention does not meet demand for {0}".format(ALLOWED_EXTENSIONS)}

        res_list = []
        ext_format = "png"
        if file and allowed_file(file.filename):
            tmp_folder = os.path.join(tmp_folder, tools.generate_unique_id())
            file_path_list = tools.preprocess_images(file, tmp_folder, ext_format)
            app.logger.info("post file: {}\tnum: {}".format(file.filename, len(file_path_list)))
            # start posting to web with temp images
            for image in file_path_list:
                res_dict = tools.post_web(image, bill_model=bill_model, url=web_url,
                                          text_angle=text_angle, text_line=text_line)
                try:
                    app.logger.info("time take: {}s\t state: {}".format(res_dict['timeTake'], res_dict['state']))
                except Exception:
                    app.logger.error(traceback.format_exc())
                res_list.append(res_dict)
            if tools.dir_nonempty(tmp_folder):
                tools.empty_dir(tmp_folder, parent=True)
        return res_list


######*******************************************************************************************######
######*******************************************************************************************######
######*******************************************************************************************######
api.add_resource(Hello, '/')
api.add_resource(Image2Text, '/image2text')

if __name__ == '__main__':
    if not debug_mode:
        http_server = WSGIServer(('0.0.0.0', 9999), app)
        http_server.serve_forever()
    else:
        app.run(host='0.0.0.0', port='9999', debug=True)
