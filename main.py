from ObjectTracking.deep_sort_tracker import DeepSortTracker
from threads.ImageInput.WebcamThread import UsbThread
from threads.ImageInput.VideoThread import VideoThread
from threads.Predictor.PredictorImage import PredictorImage
import cv2
import warnings
from utilities import constants
from utilities import helper
import argparse
from edgetpu.detection.engine import DetectionEngine
from PIL import Image
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.utils.data_utils import get_file
import numpy as np
import json
warnings.filterwarnings('ignore')

'''
WEBCAM_INPUT = 'cam'
def init(inputSrc):
    if inputSrc == WEBCAM_INPUT:
        # Run the webcam thread
        thread_image = WebcamThread('Webcam Thread', 1)
    else:
        thread_image = VideoThread('Video Thread', inputSrc, FPS=25.0)

    thread_image.start()
    image_data = thread_image.image_data
    # Run the COCO Model
    thread_coco = PredictorImage('coco',
                                 constants.CKPT_COCO,
                                 constants.LABELS_COCO,
                                 image_data,
                                 score_thresh=0.5,
                                 WITH_TRACKER=False)
    thread_coco.start()
    thread_coco.continue_predictor()
    # Initialize the Tracker
    tracker = DeepSortTracker()
    return tracker, thread_coco, thread_image
'''

def gstreamer_pipeline(capture_width = 1280, capture_heights = 720, display_width = 1280,
                       display_height = 720, framerate = 20, flip_method=0):
    return('nvarguscamerasrc ! '
           'video/x-raw(memory:NVMM), '
           'width=(int)%d, height=(int)%d, '
           'format=(string)NV12, framerate=(fraction)%d/1 ! '
           'nvvidconv flip-method=%d ! '
           'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
           'videoconvert ! '
           'video/x-raw, format=(string)BGR ! appsink' %
           (capture_width, capture_height, framerate, flip_method, display_width, display_height))


def init(input_src, faceModel, emotionModel):
    # initiate camera feed
    if input_src == 'csi':
        stream_thread = CsiThread('Csi Thread', gstreamer_pipeline())
    if input_src == 'usb':
        stream_thread = UsbThread('USB Thread', 0)
    
    stream_thread.start()
    
    # initialise models
    faceEngine = DetectionEngine(faceModel)
    global emotionList 
    emotionList = ['angry', 'disgust', 'scared', 'happy', 'sad', 'surprised', 'neutral']
    emotionNet = load_model(emotionModel, compile = False)
    
    # initialise tracker
    tracker = DeepSortTracker()
    return stream_thread, faceEngine, emotionNet, tracker

'''
def main(cap, faceEngine, emotionNet, tracker):
    frameName = 'Main Frame'
    print('Running a Tensorflow model with the DeepSORT Tracker')
    # Run the main loop
    while True:
        # Grab the image and convert from RGB -> BGR
        image_np = thread_image.image_data.image_np.copy()[:, :, ::-1]
        output_data = thread_coco.output_data
        output_data = tracker.run(output_data, image_np)
        image_np = helper.drawDetectedBBs(image_np.copy(),
                                          output_data,
                                          score_thresh=0.1)

        cv2.imshow(frameName, image_np)
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
'''

def main(stream, faceEngine, emotionNet, tracker):
    frameName = 'Main Frame'
    print('Running a Tensorflow model with the DeepSORT Tracker')
    # Run the main loop
    while True:
        frame = stream.image_data
        # frame = stream.image_data.image_np.copy()[:, :, ::-1]
        
        frame_pil = Image.fromarray(frame)
        faces = engine.DetectWithImage(frame_pil, threshold=0.05, 
                                       keep_aspect_ratio=False, relative_coord=False,
                                       top_k=10)
        image_np = helper.drawDetectedBBs(frame_pil.copy(), faces, score_thres=0.1)
        
        cv2.imshow(frameName, image_np)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    faceModel = './deploy_model/mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite'
    emotionModel = './deploy_model/emotion_net.hdf5'
    input_src = 'csi'
    
    stream, faceEngine, emotionNet, tracker = init(input_src, faceModel, emotionModel)
    main(stream, faceEngine, emotionNet, tracker)
