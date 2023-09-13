from core import *
import taichi as ti
import time
import numpy as np


if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    brain = Brain(name="brain_1")
    '''
    brain.load("./brain_1")

    brain.run(duration=20000, max_turn=200)

    '''
    exponent_size_n = 4
    exponent_size_m = 4
    main_array = SmallWorldArea(2 ** exponent_size_n, 2 ** exponent_size_m, alpha=2)
    visual = VisualSense(source='/Users/wuhuan/PycharmProjects/brain/data/1.png', shape=[2, 2, 3])
    visual.neuron_array = visual.create_buffer_area()
    print(visual.neuron_array)

    visual_main_conn = NeuronConnection(visual.neuron_array.name, main_array.name, [0, 2 * 2 * 3], [0, 2 * 2 * 3])

    brain.add(main_array)
    brain.add(visual.neuron_array)
    brain.add(visual)
    brain.add(visual_main_conn)

    start = time.time()
    brain.run(duration=3, max_turn=10)
    brain.save(".")

    brain2 = Brain("./brain2")
    brains = brain2.search(".")
    print(brains)
    brain2.load(brains[0])
    for child_1 in brain.children:
        child_2 = None
        for child in brain2.children:
            if child.name == child_1.name:
                child_2 = child
                break
        for member_name in dir(child_1):
            if member_name in dir(child_2):
                member_1 = getattr(child_1, member_name)
                member_2 = getattr(child_2, member_name)
                if isinstance(member_1, ti.ScalarField):
                    array_1 = member_1.to_numpy()
                    array_2 = member_2.to_numpy()
                    if np.equal(array_1.all(), array_2.all()):
                        print(f"[{member_name}]: Pass")
                    else:
                        print(f"[{member_name}]: Fail")


    #'''

