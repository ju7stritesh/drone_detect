import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

import cv2
import os
import numpy as np
import time
import sys
import datetime

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

open('logs.txt', 'w')

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())
model_path = os.path.join('snapshots', 'resnet50_csv_20_inference.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')


labels_to_names = {0: 'ignored_regions', 1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van', 6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor', 11: 'others'}

def run_detection_image(image):

    # copy to draw on
    h, w, layers = image.shape
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    image = preprocess_image(image)
    image, scale = resize_image(image)

    # process image
    start = time.time()
    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    # print("processing time: ", time.time() - start)

    # correct for image scale
    boxes /= scale
    count = 0

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
                count += 1
    if count >= 1:
        logs = open("logs.txt", "a+")
        logs.write("Vehicles Found at " + str(
            datetime.datetime.utcnow().strftime("%A, %d, %B %Y %I:%M%p")) + "with Count " + str(count) + '\n')
    draw_conv = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    cv2.putText(draw_conv, "Vehicles : " + str(count), (w-400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 1, lineType=cv2.LINE_AA)
    
    cv2.imshow("test", draw_conv)
    cv2.waitKey(10)
    #cv2.imwrite(output_path, draw_conv)
    return draw_conv

def main(image):
    (frame_height, frame_width, d) = image.shape
    out = cv2.VideoWriter('output/' + 'results' + '.avi', cv2.VideoWriter_fourcc(*'XVID'), 2, (frame_width, frame_height))
    out_frame = run_detection_image(image)
    out.write(out_frame)

if __name__ == '__main__':
    filepath = 'videos/test.mp4'
    cap = cv2.VideoCapture(filepath)
    while (True):
        ret, frame = cap.read()
        main(frame)
