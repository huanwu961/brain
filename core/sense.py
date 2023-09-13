import cv2 as cv
import numpy as np
import random
import time
import taichi as ti
from area import NeuronArea
from core.base import Base


class NeuronSense(Base):
    def __init__(self, source=0, shape=[1,], name='sense_'+str(random.randint(0, 100000)), config=None) -> None:
        super().__init__(name=name, class_name='NeuronSense', config=config)
        if config is None:
            self.source = source
            self.shape = shape
            self.dim = len(shape)
            self.size = 1
            for length in shape:
                self.size *= length
        self.neuron_array = None
        if isinstance(self.source, int):
            self.buffer = ti.field(dtype=ti.f32, shape=self.shape)

    def create_buffer_area(self):
        self.neuron_array = NeuronArea(self.size, 1, self.name+'_area')
        return self.neuron_array

    def connect(self, area):
        self.neuron_array = area
        print("VisualSense: connected")


@ti.data_oriented
class VisualSense(NeuronSense):
    def __init__(self, source=0, shape=(128, 128, 3), name="visual_"+str(random.randint(0, 100000)), config=None) -> None:
        super().__init__(source, shape, name, config=config)
        self.class_name = 'VisualSense'
        self.source_type = 'visual'
        self.mode = ''
        if isinstance(source, str):
            post_fix = source.split('.')[-1]
            print('[sense]:', source.split('.')[-1])
            if post_fix in ['png', 'jpg']:
                self.mode = 'image'
            elif post_fix in ['mp4']:
                self.mode = 'video'
        if isinstance(source, int):
            self.mode = 'webcam'

        if self.mode == 'webcam':
            self.videocapture = cv.VideoCapture(source)
            ret, frame = self.videocapture.read()
            self.buffer = ti.field(dtype=ti.f32, shape=frame.shape)
        elif self.mode == 'video':
            self.videocapture = cv.VideoCapture(source)
            ret, frame = self.videocapture.read()
            self.buffer = ti.field(dtype=ti.f32, shape=frame.shape)
            print("view")
        elif self.mode == 'image':
            self.image = cv.imread(source)
            self.buffer = ti.field(dtype=ti.f32, shape=self.image.shape)
        print("VideoCapture initialized")

    def read(self):
        frame = None
        if self.mode in ['webcam', 'video']:
            ret, frame = self.videocapture.read()
        elif self.mode == 'image':
            frame = cv.imread(self.source)

        # reshape the frame to the input size
        #print(self.shape[:2])
        #cv.imshow("pre", frame)
        #cv.imshow("frame", frame)
        s1 = time.time()
        self.buffer.from_numpy(frame)
        s2 = time.time()
        #self.neuron_array.current_state.from_numpy(frame.reshape(-1))
        self.reshape()
        s3 = time.time()
        print("buf", s2-s1)
        print("reshape", s3-s2)
        '''
        print("read: %f" % (start1 - start))
        print("resize: %f" % (s2 - start1))
        print("from_numpy: %f" % (s3 - s2))
        '''
        # print("VisualSense: read done")

    @ti.kernel
    def reshape(self):
        a = self.shape[0]
        b = self.shape[1]
        c = self.shape[2]
        a_ratio = self.buffer.shape[0] / a
        b_ratio = self.buffer.shape[1] / b
        for i in range(self.size):
            x = int(i / (b*c))
            y = int((i % (b*c)) / c)
            z = (i % (b*c)) % c
            # sampling as reshape
            self.neuron_array.current_state[i] = self.buffer[int(x*a_ratio), int(y*b_ratio), z] / 255
