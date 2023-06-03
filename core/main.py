import cv2 as cv
import taichi as ti
from area import SmallWorldArea
from sense import VisualSense
from connection import NeuronConnection
from brain import Brain
import os

if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    brain = Brain(name="brain_1")
    '''
    brain.load("./brain_1")
    brain.run()
    '''
    exponent_size_n = 23
    exponent_size_m = 4
    main_array = SmallWorldArea(2**exponent_size_n, 2**exponent_size_m, alpha=2)
    visual = VisualSense(source=0, shape=(512, 512, 3))

    visual_main_conn = NeuronConnection(visual.nueron_array, main_array, [0, 512*512*3], [0, 512*512*3])
    
    brain.add(main_array)
    brain.add(visual.nueron_array)
    brain.add(visual)
    brain.add(visual_main_conn)

    brain.init()

    for conn in brain.connections:
        conn.connect()
    
    counter = 0
    sleep = -1
    
    while True:
        counter += 1
        print(counter, sleep)
        if counter % 100 == 0:
            sleep *= -1
        if sleep == -1:
            for sense in brain.senses:
                sense.read()
        for connection in brain.connections:
            if sleep == -1:
                connection.connection_update()
            connection.view_update()
        for area in brain.areas:
            area.update()
        connection = brain.connections[0]

        in_frame, out_frame = connection.view_connection([512, 512, 3], [512, 512, 3])
        cv.imshow("perception", in_frame)
        cv.imshow("imagination", out_frame)
        cv.waitKey(1)

    