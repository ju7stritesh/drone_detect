from StreamReader import StreamReader
from Recognizer import Recognizer
from StreamShower import StreamShower
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process input Drone Image Detector')
    parser.add_argument('-u', '--url', action='store', type=str, default='videos/test.mp4',
                        help='Give the URL of input stream, or the Filename')

    args = parser.parse_args()
    url = args.url

    reader = StreamReader(url)
    recognizer = Recognizer(reader)
    recognizer.start_recognizing()
    shower = StreamShower('Processed', reader.frame_width, reader.frame_height)

    while True:
        ''' Dummy call for now '''
        reader.get_next_image()
        if recognizer.output_image is not None:
            shower.display_image(recognizer.output_image, recognizer.running_count)
        else:
            print('Empty image. Cannot show')

    #reader.start_reader()
    #frame_width = int(cap.get(3))
    #frame_height = int(cap.get(4))
    #out = cv2.VideoWriter('output/' + filename + '.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))


    # while(True):
    #     # Capture frame-by-frame
    #
    #     ret, frame = cap.read()
    #     print (ret)
    #     if ret is False:
    #         break
    #     out_frame = run_detection_image(frame)
    #     out.write(out_frame)
