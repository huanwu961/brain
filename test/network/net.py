import taichi as ti

ti.init(arch=ti.gpu)

@ti.kernel
def init_1(alpha: int):
    for i in range(n):
        rand = ti.random(float)
        out = int(1 / ti.pow(rand, alpha))
        for j in range(m):
            x[i, (out + j) % n] = 1
    for i, j in ti.ndrange(n, n):
        if x[i, j] != 1:
            x[i, j] = 100000000

@ti.kernel
def init_2():
    for i, j in ti.ndrange(n, m):
        rand = ti.random(float)
        out = int(rand * n)
        x[i, out] = 1
    for i, j in ti.ndrange(n, n):
        if x[i, j] != 1:
            x[i, j] = 100000000

@ti.kernel
def calc_shortest_path():
    for I in ti.grouped(x):
        for k in range(n):
            if x[I[0], I[1]] > x[I[0], k] + x[k, I[1]]:
                x[I[0], I[1]] = x[I[0], k] + x[k, I[1]]

@ti.kernel
def sum():
    for I in ti.grouped(x):
        if x[I] < 100000000:
            y[0] += x[I]

m = 10
n = 5000
shape = (n, n)
x = ti.field(dtype=ti.i32, shape=shape)
y = ti.field(dtype=ti.f32, shape=1)
#init_1(2)
init_2()
calc_shortest_path()
sum()
print(y)
