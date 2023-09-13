'''
This file is used to test the performence of different memory access ways.
'''

import taichi as ti
ti.init(arch=ti.gpu, kernel_profiler=True)

@ti.kernel
def func_init():
    for i in range(size):
        out1[i] = int(ti.random(float) * size)

@ti.kernel
def func():
    for i in range(size):
        y = int(ti.random(float) * size)
        for j in range(m):
            z = int(ti.random(float) * size)
            x[y + j] = y
            
@ti.kernel
def func_init_2():
    for i in range(size):
        r = ti.random(float)
        s = ti.random(float)
        sign = -1
        if s < 0.5:
            sign = 1
        pre_shift = int(sign * (int(1 / (r*r))%(size//2)))
        out3[i] = pre_shift

@ti.kernel
def small_world():
    shift = m//2
    for i, j in ti.ndrange(size, m):
        out = (out3[i] + j - shift) % size
        x[out] = int(ti.random(float) * size)

@ti.kernel
def random_init():
    for i, j in ti.ndrange(size, m):
        out2[i, j] = int(ti.random(float) * size)

@ti.kernel
def random():
    for i, j in ti.ndrange(size, m):
        out = out2[i, j]
        x[out] = int(ti.random(float) * size)

size = 2**18
m = 2**8
x = ti.field(dtype=ti.f32, shape=(size,))
out1 = ti.field(dtype=ti.i32, shape=(size,))
out2 = ti.field(dtype=ti.i32, shape=(size, m))
out3 = ti.field(dtype=ti.i32, shape=(size,))

random_init()
func_init()
func_init_2()

for i in range(5):
    func()

for i in range(5):
    random()
    
for i in range(5):
    small_world()

ti.profiler.print_kernel_profiler_info()
