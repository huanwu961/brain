import cv2 as cv
import random
import time
import taichi as ti
from area import NeuronArea


class NeuronSense:
    def __init__(self, source, shape, name) -> None:
        self.name = name
        self.type = 'sense'
        self.shape = shape
        self.source = source
        self.dim = len(shape)
        self.size = 1
        for length in self.shape:
            self.size *= length
        self.neuron_array = None


@ti.data_oriented
class VisualSense(NeuronSense):
    def __init__(self, source, shape, name="visual_"+str(random.randint(0, 100000))) -> None:
        super().__init__(source, shape, name)
        self.source_type = 'visual'
        self.videocapture = cv.VideoCapture(source)
        print("VideoCapture initialized")
    
    def read(self):
        start = time.time()
        ret, frame = self.videocapture.read()
        start1 = time.time()
        # frame = frame.astype(np.float32)
        # reshape the frame to the input size
        frame = cv.resize(frame, self.shape[:2])
        s2 = time.time()
        
        # self.neuron_array.current_state.from_numpy(frame.reshape(-1))
        s3 = time.time()
        print("read: %f" % (start1 - start))
        print("resize: %f" % (s2 - start1))
        print("from_numpy: %f" % (s3 - s2))
        print("VisualSense: read done")
        
    def connect(self, area):
        self.neuron_array = area
        print("VisualSense: connected")
        
    def init(self):
        self.neuron_array = NeuronArea(self.size, 1, self.name)