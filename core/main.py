from core import *
import taichi as ti

if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    '''
    n_exponent = 19
    m_exponent = 9

    brain = Brain(name='brain', root="../data")
    area = SmallWorldArea(n=2**n_exponent, m=2**m_exponent, name="area")
    sensor = VisualSense(name='visual')
    connection = NeuronConnection(
        in_name="visual",
        out_name="area",
        in_pos=[0, sensor.size],
        out_pos=[0, sensor.size]
    )

    brain.add(area)
    brain.add(sensor)
    brain.add(connection)

    brain.run(duration=10, max_turn=10)
    brain.save("../data")
    '''
    brain = Brain()
    brain.load(root="../data/brain")
    brain.run(duration=10, max_turn=10)
    #'''
