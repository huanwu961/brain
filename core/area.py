import taichi as ti
import random
from base import Base


@ti.data_oriented
class NeuronArea(Base):
    def __init__(self, n=1, m=1, name="default"):
        super().__init__(name, 'NeuronArea')
        self.n = n  # number of neurons
        self.m = m  # output size
        self.topology = {}
        print("NeuronArea %s initialized with n=%d, m=%d" % (name, n, m))

        self.current_state = ti.field(dtype=ti.f32, shape=n)
        self.cumulative_state = ti.field(dtype=ti.f32, shape=n)
        self.cumulative_weight = ti.field(dtype=ti.f32, shape=n)
        self.last_state = ti.field(dtype=ti.f32, shape=n)
        
        self.output_position = ti.field(dtype=ti.i32, shape=n)
        self.weight = ti.field(dtype=ti.f32, shape=(n, m))

        self.weight.fill(0.5)

    @ti.kernel
    def update_state(self):
        for i in ti.ndrange(self.n):
            self.last_state[i] = self.current_state[i]
            self.cumulative_state[i] = 0
            self.cumulative_weight[i] = 0
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
                w = self.weight[i, j]
                if w > 0.5:
                    dw = 1 - w
                else:
                    dw = w
                self.weight[i, j] += correlation * product_x * dw
            
    def update(self):
        self.update_state()
        print("%s: update_state done" % self.name)
        self.update_weight()
        print("%s: update_weight done" % self.name)


@ti.data_oriented
class SmallWorldArea(NeuronArea):
    def __init__(self, n, m, alpha, name="small_world_"+str(random.randint(0, 100000))):
        super().__init__(n, m, name)
        self.alpha = alpha
        self.topology["type"] = "small_world"
        self.topology['alpha'] = self.alpha
        print("start init topology...")
        self.init_topology()
        print("finished init topology")
        
    @ti.kernel
    def init_topology(self):
        for i in ti.ndrange(self.n):
            r = ti.random(float)
            s = ti.random(float)
            sign = -1
            if s > 0.5:
                sign = 1
            random_shift = int(r ** (-self.alpha)) * sign
            self.output_position[i] = i - self.m // 2 + random_shift
