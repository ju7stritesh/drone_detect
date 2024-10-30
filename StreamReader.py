import threading
import time
import logging
import cv2
from threading import Lock

th_lock = Lock()

#FILEPATH = 'videos/death_circle_output.mp4'

class StreamReader:
    def __init__(self, source):
        self.logger = logging.getLogger(__name__)

        self.latest_image = None
        self.new_image = threading.Event()
        if source.isdigit():
            self.videosource = int(source)
        else:
            self.videosource = source

        self.video_delay = 0
        self.consecutive_wait_timeouts = 0

        self.frame_good = True
        self.cv2_capture = None
        self.cam_running = True
        self.keep_running = True
        self.filereader = True
        self.frame_width = 0
        self.frame_height = 0

        #self._start_reader()

    def stop(self):
        self.keep_running = False

    def start_reader(self):
        #Initialize the Capture
        self.cap = cv2.VideoCapture(self.videosource)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))

        #TODO: We may need to add the check of incoming stream ..etc
        try:
            print('Starting the reader thread')
            time.sleep(5)
            self.reader_thread = threading.Thread(target=self._reader_thread, name='streamreader')
            self.reader_thread.start()
        except Exception as ex:
            self.logger.error("Cannot start Stream reader thread", str(ex))

    def get_next_image(self):
        if not self.new_image.wait(2.0):  # 2.0 is 2 second wait timeout if no frames come in
            self.consecutive_wait_timeouts += 1
            if self.consecutive_wait_timeouts > 9:
                self.logger.warning(str(self.consecutive_wait_timeouts) + ' consecutive wait timeouts.')
                self.consecutive_wait_timeouts = 0
            self.frame_good = False
        else:
            self.consecutive_wait_timeouts = 0
        self.new_image.clear()
        #print('get_next_image. Returning ', self.latest_image)
        th_lock.acquire()
        cv_frame = self.latest_image
        th_lock.release()
        return cv_frame

    def _reader_thread(self):
        while self.keep_running:
            if self.filereader:
                #Fixme: Remove this sleep once we move to RTSP
                time.sleep(1)
                try:
                    ret, frame = self.cap.read()
                    if ret is False:
                        self.logger.error('Cannot read the image ')
                        break
                    else:
                        th_lock.acquire()
                        self.latest_image = frame
                        th_lock.release()

                        self.__got_goodframe()
                except Exception as ex:
                    # Just keeps trying
                    self.logger.error('Exception in reading image ' + str(ex))
        self.latest_image = None
        self.logger.error('Image Reader thread thread stopping')

    def __got_goodframe(self):
        if self.latest_image is not None:
            self.new_image.set()

            self.frame_good = True
            self.cam_running = True
        else:
            self.logger.error('got_good_frame but latest_image is None.')




