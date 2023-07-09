import taichi as ti
import random
from core.base import Base


@ti.data_oriented
class NeuronArea(Base):
    def __init__(self, n=1, m=1, name="default", config=None):
        super().__init__(name, class_name='NeuronArea', config=config)
        if config is None:
            self.n = n  # number of neurons
            self.m = m  # output size
        self.topology = {}

        print("NeuronArea %s initialized with n=%d, m=%d" % (name, self.n, self.m))
        self.current_state = ti.field(dtype=ti.f32, shape=self.n)
        self.cumulative_state = ti.field(dtype=ti.f32, shape=self.n)
        self.cumulative_weight = ti.field(dtype=ti.f32, shape=self.n)
        self.last_state = ti.field(dtype=ti.f32, shape=self.n)
        
        self.output_position = ti.field(dtype=ti.i32, shape=self.n)
        self.weight = ti.field(dtype=ti.f32, shape=(self.n, self.m))

        self.weight.fill(0.5)

    @ti.kernel
    def update_state(self):
        for i in range(self.n):
            self.last_state[i] = self.current_state[i]
        for i, j in ti.ndrange(self.n, self.m):
            tar = (self.output_position[i] + j) % self.n
            if tar == i:
                continue
            self.cumulative_state[tar] += self.weight[i, j] * self.current_state[i]
            self.cumulative_weight[tar] += self.weight[i, j]
        for i in range(self.n):
            if self.cumulative_weight[i] == 0:
                self.current_state[i] = 0
            else:
                self.current_state[i] = self.cumulative_state[i] / self.cumulative_weight[i]

    @ti.kernel
    def update_weight(self):
        for i in ti.ndrange(self.n):
            for j in ti.ndrange(self.m):
                tar = (self.output_position[i] + j) % self.n
                if tar == i:
                    continue
                correlation = (1 - 2*self.last_state[i]) * (1 - 2*self.current_state[tar])
                product_x = self.last_state[i] * self.current_state[tar]
                dw = self.weight[i, j]
                if dw > 0.5:
                    dw = 1 - dw
                self.weight[i, j] += correlation * product_x * dw

    @ti.kernel
    def update_weight_2(self):
        for i in ti.ndrange(self.n):
            weight_sum = 0
            for j in range(self.m):
                tar = (self.output_position[i] + j) % self.n
                if tar == i:
                    continue
                self.weight[i, j] += self.last_state * self.current_state[tar]
                weight_sum += self.weight[i, j]
            for j in range(self.m):
                self.weight[i, j] /= weight_sum

    def update(self):
        self.update_state()
        print("%s: update_state done" % self.name)
        self.update_weight()
        print("%s: update_weight done" % self.name)

    @ti.kernel
    def clear_cumulative(self):
        for i in range(self.n):
            self.cumulative_state[i] = 0
            self.cumulative_weight[i] = 0



@ti.data_oriented
class SmallWorldArea(NeuronArea):
    def __init__(self, n=1, m=1, alpha=2, name="small_world_"+str(random.randint(0, 100000)), config=None):
        super().__init__(n, m, name, config=config)
        self.class_name = 'SmallWorldArea'
        self.topology["type"] = "small_world"
        self.alpha = alpha
        self.topology['alpha'] = self.alpha
        print("start init topology...")
        self.init_topology()
        print("finished init topology")
        
    @ti.kernel
    def init_topology(self):
        for i in range(self.n):
            r = ti.random(float)
            s = ti.random(float)
            sign = -1
            if s > 0.5:
                sign = 1
            random_shift = int(1 / r * r) * sign
            self.output_position[i] = i - self.m // 2 + random_shift
