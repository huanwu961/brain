from core import *
import taichi as ti
import time
import numpy as np
import cv2


if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    brain = Brain(name="brain_1")
    '''
    brain.load("./brain_1")
    brain.run(duration=20000, max_turn=200)
    '''
    exponent_size_n = 20
    exponent_size_m = 8
    main_array = SmallWorldArea(2 ** exponent_size_n, 2 ** exponent_size_m, alpha=2)
    # visual = VisualSense(source='/Users/wuhuan/PycharmProjects/brain/data/1.png', shape=[720, 1280, 3])
    visual = VisualSense(source=0, shape=[360, 640, 3])
    visual.neuron_array = visual.create_buffer_area()

    visual_main_conn = NeuronConnection(visual.neuron_array.name, main_array.name, [0, 360 * 640 * 3], [0, 360 * 640 * 3])

    brain.add(main_array)
    brain.add(visual.neuron_array)
    brain.add(visual)
    brain.add(visual_main_conn)

    start = time.time()

    times = {}
    states = {}
    brain.prepare()
    for i in range(200):
        print("turn", i)
        start_t = time.time()
        for sensor in brain.senses:
            start = time.time()
            sensor.read()
            end = time.time()
            times[sensor.name] = end - start
            cv2.imshow(sensor.name, ((sensor.neuron_array.current_state.to_numpy()[:360*640*3]).reshape([360, 640, 3])))
        for connection in brain.connections:
            start = time.time()
            connection.update()
            end = time.time()
            times[connection.name] = end - start
        for area in brain.areas:
            start = time.time()
            area.update()
            end = time.time()
            times[area.name] = end - start
            states[area.name] = ((area.current_state.to_numpy()[:360*640*3]*255).reshape([360, 640, 3])).astype(np.uint8)
            cv2.imshow(area.name, states[area.name])
        for action in brain.actions:
            start = time.time()
            action.act()
            end = time.time()
            times[action.name] = end - start
        print(main_array.current_state.to_numpy()[:2000])
        end_t = time.time()
        times["total"] = end_t - start_t
        print(times)

        cv2.waitKey(1)

        for area in brain.areas:
            area.clear_cumulative()

    main_array.random_pertubation()
    for i in range(100):
        main_array.update_state()
        print(main_array.current_state.to_numpy()[200000:200300])
        main_array.clear_cumulative()
        print("turn", i)
        cv2.imshow("main", (main_array.current_state.to_numpy()[:360*640*3]*255).reshape([360, 640, 3]).astype(np.uint8))
        cv2.waitKey(1)
        '''
        for child in brain.children:
            # print(f"[{child.name}]: logging...")
            child.monitor()
        '''



# [0.16919511 0.16400937 0.15739618 0.1070116 ]
    #'''
