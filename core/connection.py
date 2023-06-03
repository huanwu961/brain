import taichi as ti
import numpy as np
from base import Base
import utils


@ti.data_oriented
class NeuronConnection(Base):
    def __init__(self, in_array=None, out_array=None, in_pos=(0, 1), out_pos=(0, 1), weight=1, m=1,
                 conn_type="continuous"):
        super().__init__('empty', 'NeuronConnection')
        self.base = False
        if in_array is not None and out_array is not None:
            print(in_array, out_array)
            self.name = "%s->%s" % (in_array.name, out_array.name)
        self.in_name = ''
        self.out_name = ''
        self.in_array = in_array
        self.out_array = out_array
        self.in_pos = in_pos
        self.out_pos = out_pos
        self.weight = weight
        self.conn_type = conn_type
        self.m = m
        self.in_length = in_pos[1] - in_pos[0]
        self.out_length = out_pos[1] - out_pos[0]
        self.output_position = ti.field(dtype=ti.i32, shape=self.in_length)
        self.out_array_state = ti.field(dtype=ti.f32, shape=self.out_length)
        self.in_array_state = ti.field(dtype=ti.f32, shape=self.out_length)
        self.kl_div = 0
        self.init_topology()
        print("Connection %s initialized with type=%s, weight=%f, m=%d" % (self.name, self.conn_type, self.weight, self.m))
        print("ready to connect, waiting for target area...")
        if in_array is not None and out_array is not None:
            self.connect(in_array, out_array)

    def connect(self, area1, area2):
        self.in_array = area1
        self.out_array = area2
        self.in_name = area1.name
        self.out_name = area2.name
        print("Connection finished.")
        
    @ti.kernel
    def init_topology(self):
        if self.conn_type == "continuous":
            for i in range(self.in_length):
                shift = int(i / self.in_length * self.out_length)
                self.output_position[i] = shift + self.out_pos[0]

        print("Add connection type %s from [%d, %d] to [%d, %d]" %
              (self.conn_type, self.in_pos[0], self.in_pos[1], self.out_pos[0], self.out_pos[1]))
    
    @ti.kernel
    def update(self):
        for i, j in ti.ndrange(self.in_length, self.m):
            out = (self.output_position[i] + j) % self.out_length
            self.out_array.cumulative_state[out] += self.in_array.current_state[self.in_pos[0]+i] * self.weight
            self.out_array.cumulative_weight[out] += self.weight
            self.in_array_state[j] = self.out_array.cumulative_state[out]

    def view_connection(self, in_shape, out_shape):
        in_frame = self.in_array.current_state.to_numpy()
        in_frame = in_frame.reshape(in_shape) * 255
        out_frame = self.out_array_state.to_numpy()
        out_frame = out_frame.reshape(out_shape) * 255
        return in_frame, out_frame

    @ti.kernel
    def calc_kl_divergence(self):
        self.kl_div = 0

        # calculate softmax of in_array_state and out_array_state
        e_in_sum = 0
        e_out_sum = 0
        for I in ti.grouped(self.out_array_state):
            # separate the pure generated information from outside input
            self.out_array_state[I] = ti.math.e ** ((self.in_array.cumulate_state[I] - self.in_array_state[I] * self.weight) / \
                                      (self.out_array.cumulate_weight[I] - self.weight))
            self.in_array_state[I] = ti.math.e ** self.in_array_state[I]
            e_in_sum += self.in_array_state[I]
            e_out_sum += self.out_array_state[I]

        for I in ti.grouped(self.out_array_state):
            self.in_array_state[I] /= e_in_sum
            self.out_array_state[I] /= e_out_sum

        # calculate kl_divergence
        # since we are using out_array_state to simulate the distribution of in_array_state,
        # then KL(p|q) will be KL(in|out) = -in * log(in/out)
        for I in ti.grouped(self.out_array_state):
            self.kl_div += self.in_array_state[I] * \
             ti.log(self.in_array_state[I] / self.out_array_state[I])

