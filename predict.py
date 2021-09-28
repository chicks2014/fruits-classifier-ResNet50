#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 18:45:05 2020

@author: chetanhirapara
"""

import numpy as np
# from keras.models import load_model
from keras.preprocessing import image

# from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import logging
logging.basicConfig(filename='/opt/python/log/my.log', level=logging.DEBUG)

class classification:
    def __init__(self,filename):
        self.filename =filename


    def prediction(self):
        logging.log(msg="prediction func. called", level=logging.DEBUG)
        # load model
        model = load_model('resNet50_fruits_recognition.model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224, 224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)
        logging.log(msg="model.predict called", level=logging.DEBUG)

        classes = ['apple_6', 'apple_braeburn_1', 'apple_crimson_snow_1', 'apple_golden_1', 'apple_golden_2', 'apple_golden_3', 'apple_granny_smith_1', 'apple_hit_1', 'apple_pink_lady_1', 'apple_red_1', 'apple_red_2', 'apple_red_3', 'apple_red_delicios_1', 'apple_red_yellow_1', 'apple_rotten_1', 'cabbage_white_1', 'carrot_1', 'cucumber_1', 'cucumber_3', 'eggplant_violet_1', 'pear_1', 'pear_3', 'zucchini_1', 'zucchini_dark_1']

        pred = classes[np.argmax(result)]
        logging.log(msg=f"model predict result: {pred}", level=logging.DEBUG)
        return [{ "image" : pred}]


