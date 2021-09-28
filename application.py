from flask import Flask, request, jsonify, render_template
import os
from flask_cors import CORS, cross_origin
import logging
logging.basicConfig(filename='/opt/python/log/my.log', level=logging.DEBUG)
from custom_utils.utils import decodeImage
from predict import classification

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

application = Flask(__name__)
CORS(application)


# @cross_origin()
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = classification(self.filename)


@application.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')


@application.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    logging.log(msg="predictRoute called", level=logging.DEBUG)
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.prediction()
    return jsonify(result)


# define app
clApp = ClientApp()
if __name__ == "__main__":
    logging.log(msg="main func called", level=logging.DEBUG)
    application.run(debug=True)
