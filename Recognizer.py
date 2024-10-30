import threading
import logging

import keras
# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import tensorflow as tf

import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

global graph
graph = tf.get_default_graph()
labels_to_names = {0: 'ignored_regions', 1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van', 6: 'truck',
                   7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11: 'others'}

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('snapshots', 'resnet50_csv_20_inference.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

class Recognizer:
    def __init__(self, reader) :
        self.logger = logging.getLogger(__name__)
        self.streamreader = reader
        self.output_image = None

    def start_recognizing(self):
        recognizer_thread = threading.Thread(target=self.__get_recognize_image, name='recognizer')
        recognizer_thread.start()
        self.streamreader.start_reader()

    def __get_recognize_image(self):
        while True:
            start = time.time()
            cv_frame = self.streamreader.get_next_image()
            self.logger.info("get image time: " + str(time.time() - start))
            if cv_frame is not None:
                self.run_detection_image(cv_frame)
            else:
                self.logger.error('Empty image frame. Skipping')

    def run_detection_image(self, image):
        h, w, layers = image.shape
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)

        # process image
        start = time.time()
        with graph.as_default():
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)

        # correct for image scale
        boxes /= scale
        self.running_count = 0

        # visualize detections
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break

            if score > 0.7:
                if label == 4 or label == 5 or label == 6 or label == 9 or label == 10:
                    color = label_color(label)

                    b = box.astype(int)
                    draw_box(draw, b, color=color)

                    caption = "{} {:.3f}".format(labels_to_names[label], score)
                    draw_caption(draw, b, caption)
                    self.running_count += 1

        draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        cv2.putText(draw_conv, "Vehicles : " + str(self.running_count), (w-400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, lineType=cv2.LINE_AA)
        self.output_image = draw_conv
        return draw_conv

