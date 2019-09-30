# -*- coding: utf-8 -*-

import cv2
from threads.ImageInput.AbstractImageInputThread \ 
    import AbstractImageInputThread

class CsiThread(AbstractImageInputThread):
    def __init__(self, name, gstreamer_init):
        super().init(name, IMAGE_WIDTH, IMAGE_HEIGHT)
        self.cap = self.init_input(IMAGE_WIDTH, IMAGE_HEIGHT, gstreamer_init)
    
    def init_input(self, IMAGE_WIDTH, IMAGE_HEIGHT, gstreamer_init):
        cap = cv2.VideoCapture(gstreamer_init, cv2.CAP_GSTREAMER)
        assert cap.isOPened(), 'Could not open CSI Camera.'
        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)
        
    def stop(self):
        super().stop()
        self.cap.release()
        
        