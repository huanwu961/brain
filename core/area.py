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

        self.field_builder = ti.FieldsBuilder()

        self.current_state = ti.field(dtype=ti.f32)
        self.cumulative_state = ti.field(dtype=ti.f32)
        self.cumulative_weight = ti.field(dtype=ti.f32)
        self.last_state = ti.field(dtype=ti.f32)
        self.output_position = ti.field(dtype=ti.i32)

        self.weight = ti.Vector.field(dtype=ti.f32, n=self.m)
        self.last_weight = ti.Vector.field(dtype=ti.f32, n=self.m)

        self.weight_diff = ti.field(dtype=ti.f32)

        self.field_builder.dense(ti.i, self.n).place(self.current_state,
                                                     self.last_state,
                                                     self.cumulative_state,
                                                     self.cumulative_weight,
                                                     self.output_position
                                                     )
        self.field_builder.dense(ti.i, self.n).place(self.weight)
        self.field_builder.dense(ti.i, self.n).place(self.last_weight)
        self.field_builder.dense(ti.i, 1).place(self.weight_diff)
        self.field_builder.finalize()
        self.init()

    @ti.kernel
    def init(self):
        self.weight.fill(0.5)
        for i in range(self.n):
            self.output_position[i] = i
        print("NeuronArea %s initialized with n=%d, m=%d" % (self.name, self.n, self.m))

    @ti.kernel
    def update_state(self):
        # set the last state to current state
        for i in range(self.n):
            self.last_state[i] = self.current_state[i]
        # gather input from others and store them in cumulative_state
        for i, j in ti.ndrange(self.n, self.m):
            tar = (self.output_position[i] + j) % self.n
            if tar == i:
                continue
            self.cumulative_state[tar] += self.weight[i][j] * self.current_state[i]
            self.cumulative_weight[tar] += self.weight[i][j]
        for i in range(self.n):
            if self.cumulative_weight[i] == 0:
                self.cumulative_weight[i] = 1
            else:
                self.current_state[i] = 0.3*self.last_state[i] + 0.7*self.cumulative_state[i] / self.cumulative_weight[i]

    @ti.kernel
    def update_weight(self):
        for i in ti.ndrange(self.n):
            for j in ti.ndrange(self.m):
                tar = (self.output_position[i] + j) % self.n
                if tar == i:
                    continue
                correlation = (1 - 2*self.last_state[i]) * (1 - 2*self.current_state[tar])
                product_x = self.last_state[i] * self.current_state[tar]
                dw = self.weight[i][j]
                if dw > 0.5:
                    dw = 1 - dw
                self.weight[i][j] += 100*correlation * product_x * dw

    @ti.kernel
    def update_weight_2(self):
        for i in ti.ndrange(self.n):
            weight_sum = 0.0
            for j in range(self.m):
                tar = (self.output_position[i] + j) % self.n
                if tar == i:
                    continue
                self.weight[i][j] += ti.pow(2, self.last_state[tar]) * ti.pow(2, self.current_state[i])
                weight_sum += self.weight[i][j]
            for j in range(self.m):
                self.weight[i][j] /= weight_sum

    @ti.kernel
    def update_weight_3(self):
        for i, j in ti.ndrange(self.n, self.m):
            tar = (self.output_position[i] + j) % self.n
            if tar == i:
                continue
            self.weight[i][j] += 100 * (self.last_state[tar] * self.current_state[i] - self.current_state[tar] * self.last_state[i])

    def update(self):
        self.update_state()
        self.update_weight_2()

    @ti.kernel
    def clear_cumulative(self):
        for i in range(self.n):
            self.cumulative_state[i] = 0
            self.cumulative_weight[i] = 0
            
    @ti.kernel
    def clear_half_state(self):
        for i in range(int(self.n/16)):
            self.current_state[i] = 0.5
            
    @ti.kernel
    def random_pertubation(self):
        for i in range(int(self.n)):
            self.current_state[i] += 0.5 * (ti.random(float) - 0.5)

    @ti.kernel
    def monitor(self):
        pass


@ti.data_oriented
class SmallWorldArea(NeuronArea):
    def __init__(self, n=1, m=1, alpha=2, name="small_world_"+str(random.randint(0, 100000)), config=None):
        super().__init__(n, m, name, config=config)
        self.class_name = 'SmallWorldArea'
        self.topology["type"] = "small_world"
        self.alpha = alpha
        self.output_shift_sum = ti.field(dtype=ti.int32, shape=(1))
        self.topology['alpha'] = self.alpha
        self.init()
        self.calc_output_shift()

    @ti.kernel
    def init(self):
        for i in range(self.n):
            r = ti.random(float)
            s = ti.random(float)
            sign = -1
            if s > 0.5:
                sign = 1
            random_shift = 5*int(1 / r * r) * sign
            self.output_position[i] = int(r * self.n)  #(i - self.m // 2 + random_shift) % self.n

    @ti.kernel
    def calc_output_shift(self):
        for i in range(self.n):
            self.output_shift_sum[0] += ti.abs(self.output_position[i] - i)
