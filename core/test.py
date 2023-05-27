import data
import brain
import area
import connection
import sense
import taichi as ti

if __name__ == '__main__':
    ti.init(arch=ti.gpu)
    data_manager = data.Data(root_path=".")
    '''
    brain1 = brain.Brain(name="brain_1")
    area1 = area.SmallWorldArea(2**23, 2**4, alpha=2)
    sense1 = sense.VisualSense(source=0, shape=(512, 512, 3))
    connection1 = connection.NeuronConnection(sense1.neuron_array, area1, [0, 512*512*3], [0, 512*512*3])
    
    brain1.add_area(area1)
    brain1.add_sense(sense1)
    brain1.add_connection(connection1)
    
    data_manager.save(brain1)
    '''
    brain1 = data_manager.load("brain_1")

    brain1.run()
    