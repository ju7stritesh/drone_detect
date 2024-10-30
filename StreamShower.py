import cv2
import numpy as np

OUT_FILE = 'output/test'

class StreamShower:
    def __init__(self, name, frame_width, frame_height):
        self.name = name
        self.fw = frame_width
        self.fh = frame_height
        print('Frame width and height are ', frame_width, frame_height)
        ''' Open a file writer'''
        filename = OUT_FILE
        self.out = cv2.VideoWriter(filename + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                              (frame_width, frame_height))

    def display_images_side_by_side(self, image_right, image_left):
        if image_right is None:
            return
        if image_left is None:
            image_left = image_right

        height, width, channels = image_right.shape
        wide_image = np.zeros((height, width * 2, channels), np.dtype('B'))

        wide_image[:height, :width, :channels] = image_right
        wide_image[:height, width:width*2, :3] = image_left

        show_image = cv2.cvtColor(wide_image, cv2.COLOR_RGB2BGR)

        cv2.imshow(self.name, show_image)
        cv2.waitKey(1)

    def display_image(self, np_image, count):
        #show_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2BGR)
        h, w, layers = np_image.shape

        cv2.putText(np_image, "Vehicles : " + str(count), (w - 400, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (255, 255, 255), 1, lineType=cv2.LINE_AA)
        show_image = np_image

        cv2.imshow(self.name, show_image)
        cv2.waitKey(10)
        self.out.write(show_image)
