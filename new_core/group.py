import json
import utils
import taichi as ti
from base import Base
import random


@ti.data_oriented
class NeuronGroup(Base):
    def __init__(self, shape=(1, 1, 1, 1), kernel=(1, 1, 1, 1), dtype=ti.f64, name=("group_" + str(int(random.randint(0, 10000))))):
        super().__init__(name=name, class_name="NeuronGroup")
        self.shape = shape
        self.kernel = kernel
        self.dtype = dtype
        
        # load protocol config
        with open("../protocol/default.json", "r") as f:
            self.protocol = json.load(f)
        
        # calculate feature length for each neuron according to protocol
        self.feature_len = 0
        for feature in self.protocol:
            self.feature_len += self.protocol[feature][1]
        
        # init array
        self.array = ti.Vector.field(n=self.feature_len, dtype=dtype, shape=shape)

    def init(self):
        pass

    def update(self):
        pass


class NeuronArea(NeuronGroup):
    def __init__(self, shape=(1, 1, 1, 1), kernel=(1, 1, 1, 1), dtype=ti.f64, name=utils.name("area")):
        super().__init__(shape=shape, kernel=kernel, dtype=dtype, name=name)
        with open("../protocol/default.json") as f:
            self.protocol = json.load(f)

    def update(self):
        self.update_state()
        self.update_weight()

    @ti.kernel
    def update_state(self):
        # load hyper-param to local var
        output_position = self.protocol["output_position_pos"][0]
        output_position_len = self.protocol["output_position_len"][1]
        current_state = self.protocol["current_state"][0]
        current_state_len = self.protocol["current_state"][1]
        last_state= self.protocol["last_state"][0]
        last_state_len = self.protocol["last_state"][1]
        cumulative_state = self.protocol["cumulative_state"][0]
        cumulative_state_len = self.protocol["cumulative_state"][1]
        cumulative_weight = self.protocol["cumulative_weight"][0]
        cumulative_weight_len = self.protocol["cumulative_weight"][1]
        weight_pos = self.protocol["weight"][0]
        weight_len = self.protocol["weight"][1]

        # move current state to last state
        for I in ti.grouped(self.array):
            self.array[I][last_state] = self.array[I][current_state]

        # combine the index of shape and kernel into a 8-dim vector
        total_shape = ti.Vector(8, int)
        for i in range(4):
            total_shape[i] = self.shape[i]
            total_shape[i + 4] = self.kernel[i]

        # main loop of updating state
        for index_vec in ti.ndrange(total_shape):
            '''
            This loop is performing 4-dim convolution where the position of output kernel 
            has some shift obaying the inverse polynomial with respect to the distance to 
            the center.
            
            local variables:
            index_vec (8-dim vector): tuple(current: 4-dim vector, kernel_shift: 4-dim vector)
            current_vec (4-dim vector): the vector position of current neuron
            shift_vec (4-dim vector): the vector discribing output shift
            out_pos_vec (4-dim vector): the vector position of output neuron
            '''
            # dispatch index to current neuron and its output neuron
            current_vec = ti.Vector(4, int)
            shift_vec = ti.Vector(4, int)
            out_pos_vec = ti.Vector(4, int)
            for i in range(4):
                current_vec[i] = index_vec[i]
                shift_vec[i] = index_vec[i + 4]
            # calculate the index of output neuron according to the output_position in self.array
            for i in range(4):
                out_pos_vec[i] = self.array[current_vec][int(output_position + i)] + shift_vec[i] # here output_position is stored as float
            # calculate the index of current_state and cumulative_state
            shift_int = self.vec4_to_int(shift_vec, self.kernel)
            
            self.array[out_pos_vec][cumulative_state] += self.array[current_vec][current_state] * self.array[current_vec][weight_pos + shift_int]
            self.array[out_pos_vec][cumulative_weight] += self.array[current_vec][weight_pos + shift_int]
        
        # calculate the average of cumulative_state and store it in current_state
        for I in ti.grouped(self.array):
            for i in range(weight_len):
                self.array[I][current_state] = self.array[I][cumulative_state] / self.array[I][cumulative_weight]
                
                
    def update_weight(self):
        self.naive_habbien_learning()
    

    @ti.kernel
    def naive_habbien_learning(self):
        # load hyper-param to local var
        output_position = self.protocol["output_position_pos"][0]
        output_position_len = self.protocol["output_position_len"][1]
        current_state = self.protocol["current_state"][0]
        current_state_len = self.protocol["current_state"][1]
        last_state= self.protocol["last_state"][0]
        last_state_len = self.protocol["last_state"][1]
        cumulative_state = self.protocol["cumulative_state"][0]
        cumulative_state_len = self.protocol["cumulative_state"][1]
        cumulative_weight = self.protocol["cumulative_weight"][0]
        cumulative_weight_len = self.protocol["cumulative_weight"][1]
        weight_pos = self.protocol["weight"][0]
        weight_len = self.protocol["weight"][1]

        # combine the index of shape and kernel into a 8-dim vector
        total_shape = ti.Vector(8, int)
        for i in range(4):
            total_shape[i] = self.shape[i]
            total_shape[i + 4] = self.kernel[i]
        
        # clear the cumulative_weight
        for I in ti.grouped(self.array):
            self.array[I][cumulative_weight] = 0
        
        # main loop of updating weight
        for index_vec in ti.ndrange(total_shape):
            # init three index vectors
            current_vec = ti.Vector(4, int)
            shift_vec = ti.Vector(4, int)
            out_pos_vec = ti.Vector(4, int)
            
            # dispatch index to current neuron and its output neuron
            for i in range(4):
                current_vec[i] = index_vec[i]
                shift_vec[i] = index_vec[i + 4]
            
            # calculate the index of output neuron according to the output_position in self.array
            for i in range(4):
                out_pos_vec[i] = self.array[current_vec][int(output_position + i)] + shift_vec[i]
            shift_int = self.vec4_to_int(shift_vec, self.kernel)
            
            # ---------------- naive habbien learning -----------------
            self.array[current_vec][weight_pos + shift_int] += self.array[current_vec][current_state] * self.array[out_pos_vec][last_state]
            self.array[current_vec][cumulative_weight] += self.array[current_vec][weight_pos + shift_int]
            # ---------------------------------------------------------
            
        # weight normalization
        for I in ti.grouped(self.array):
            for i in range(weight_len):
                self.array[I][weight_pos + i] /= self.array[I][cumulative_weight]
          
      
    # ---------------------- helper functions ----------------------    
    @ti.func
    def int_to_vec4(index: int, shape: ti.types.vector(4, int)) -> ti.types.vector(4, int):
        dim = 4
        vec = ti.types.vector(4, int)
        multiplier = 1
        remainder = 0
        for i in range(dim - 1):
            multiplier *= shape[i + 1]
        for i in range(dim):
            vec[i] = index // multiplier
            remainder = index % multiplier
            index = remainder
            multiplier //= shape[i + 1]
        return vec

    @ti.func
    def vec4_to_int(index: ti.types.vector(4, int), shape: ti.types.vector(4, int)) -> int:
        dim = 4
        multiplier = 1
        out = 0
        for i in range(dim - 1):
            multiplier *= shape[i + 1]
        for i in range(dim):
            out += index[i] * multiplier
            multiplier //= shape[i + 1]
        return out


if __name__ == '__main__':
    ti.init(print_ir=True, arch=ti.cpu, kernel_profiler=True, device_memory_fraction=0.7, device_memory_GB=1)
    area = NeuronArea(shape=(1, 1, 1, 1), kernel=(1, 1, 1, 1), dtype=ti.f64)
    area.save(root=".")
