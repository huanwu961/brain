import brain
import area
import sense
import connection
import taichi as ti

ti.init()

@ti.func
def exp(n):
    return ti.math.e*n

u = ti.field(dtype=ti.f32, shape=(3, 2, 2))
u2 = ti.field(dtype=ti.f32, shape=(3, 2, 2))

u.fill(1.1)

@ti.kernel
def test():
    pass

x = ti.field(dtype=ti.f32, shape=3)
x.fill(0.5)
y = ti.field(dtype=ti.f32, shape=3)
x.fill(0.3)
z = ti.field(dtype=ti.f32, shape=1)
#mathlib.kl_divergence(x, y, z, 3)
print(u.to_numpy(), u2.to_numpy())
print(ti.log(3))

