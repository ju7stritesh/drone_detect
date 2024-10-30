import os
from socket import *
from struct import unpack
import numpy
import cv2
import inference
import sys
import time


class TCPIPReader:

    def __init__(self):
        self.socket = None
        self.output_dir = '.'
        self.file_num = 1

        self.latest_image = None
        self.frame_count = 1
        self.fps = 1

    def listen(self, server_ip, server_port):
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)

    def handle_images(self):

        try:
            while True:
                (connection, addr) = self.socket.accept()
                try:
                    bs = connection.recv(8)
                    (length,) = unpack('>Q', bs)
                    data = b''
                    while len(data) < length:
                        # doing it in batches is generally better than trying
                        # to do it all in one go, so I believe.
                        to_read = length - len(data)
                        data += connection.recv(
                            4096 if to_read > 4096 else to_read)

                    # send our 0 ack
                    assert len(b'\00') == 1
                    connection.sendall(b'\00')
                except Exception as ex:
                    print(ex)
                finally:
                    connection.shutdown(SHUT_WR)
                    connection.close()

                    image = numpy.fromstring(data, dtype='uint8')
                    self.latest_image = cv2.imdecode(image, 1)
                    self.frame_count += 1
                    if self.frame_count % self.fps == 0 :
                        start_time = time.time()
                        inference.main(self.latest_image)
                        time_taken = time.time() - start_time
                        self.fps = int(time_taken * int(sys.argv[3]))
                        self.fps = int(int(sys.argv[3])/self.fps)
                        if self.fps == 0:
                            self.fps = 1

                self.file_num += 1
        except Exception as ex:
            print(ex)
        finally:
            self.close()

    def get_latest_image(self):
        return self.latest_image

    def __save_image(self, image):
        with open(os.path.join(
                self.output_dir, '%06d.jpg' % self.file_num), 'wb'
        ) as fp:
            fp.write(image)

    def close(self):
        self.socket.close()
        self.socket = None

        # could handle a bad ack here, but we'll assume it's fine.

def main():
    sp = TCPIPReader()
    sp.listen(sys.argv[1], int(sys.argv[2]))
    sp.handle_images()


if __name__ == '__main__':
    main()