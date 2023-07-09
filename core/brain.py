import random
import time
import psutil
import os
from area import SmallWorldArea
from area import NeuronArea
from action import NeuronAction
from sense import NeuronSense
from connection import NeuronConnection
from base import Base
from factory import Factory
import numpy as np


class Brain:
    def __init__(self, name="brain_"+str(random.randint(0, 1000000)), root='.'):
        self.name = name
        self.areas = []
        self.connections = []
        self.senses = []
        self.actions = []
        self.children = []
        self.info = {}
        self.root = root

    def prepare(self):
        # load areas first, preventing void connection
        for child in self.children:
            if isinstance(child, NeuronArea):
                self.areas.append(child)
                print("[prepare area]:", child.name, child.n, child.current_state.shape)

        for child in self.children:
            if isinstance(child, NeuronSense):
                self.senses.append(child)
                child.connect(self.get_neuron_area(child.name+'_area'))
                print("prepare connected")
            elif isinstance(child, NeuronAction):
                self.actions.append(child)
            elif isinstance(child, NeuronConnection):
                self.connections.append(child)
                area_1_name, area_2_name = child.name.split('->')
                print("[Debug]:", area_1_name, area_2_name)
                print("[Debug]: ", self.get_neuron_area(area_1_name), self.get_neuron_area(area_2_name))
                child.connect(self.get_neuron_area(area_1_name), self.get_neuron_area(area_2_name))

    def run(self, duration=None, max_turn=None):
        start_time = time.time()
        self.prepare()
        turn = 0
        while True:
            self.info['start'] = time.time()
            for sensor in self.senses:
                sensor.read()
            self.info['read_time'] = time.time()
            for connection in self.connections:
                print("[value]:", np.max(connection.out_array.cumulative_state.to_numpy()))
                connection.update()
                print("[value]:", np.max(connection.out_array.cumulative_state.to_numpy()))
                
            self.info['connection_time'] = time.time()
            for area in self.areas:
                print("[]:", np.max(area.cumulative_state.to_numpy()))
                area.update()
            self.info['main_update_time'] = time.time()
            for action in self.actions:
                action.act()
            for child in self.children:
                print(f"[{child.name}]: logging...")
                child.monitor()
            for area in self.areas:
                area.clear_cumulative()
            self.print_info()

            turn += 1
            if turn >= max_turn:
                self.save(root=self.root)
                print(f"training finished, [{turn} turns]")
                break
            elif time.time() - start_time > duration:
                print(f"training finished, [{duration}s]")
                break

    def get_neuron_area(self, name):
        for area in self.areas:
            if area.name == name:
                return area
        return None

    def get_neuron_connection(self, name):
        for connection in self.connections:
            if connection.name == name:
                return connection
        return None
    
    def get_neuron_sense(self, name):
        for sense in self.senses:
            if sense.name == name:
                return sense
        return None
    
    def get_neuron_action(self, name):
        for action in self.actions:
            if action.name == name:
                return action
        return None

    def print_info(self):
        print("Read time: %f" % (self.info['read_time'] - self.info['start']))
        print("Connection time: %f" % (self.info['connection_time'] - self.info['read_time']))
        print("Main update time: %f" % (self.info['main_update_time'] - self.info['connection_time']))
        self.info['memory_size'] = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        print("mem usage: %f MB" % self.info['memory_size'])

    def add(self, obj):
        self.children.append(obj)

    def save(self, root):
        brain_root = os.path.join(root, self.name)
        os.makedirs(brain_root, exist_ok=True)

        for child in self.children:
            print('[configuration]: ', child.configuration)
            child.save(brain_root)

    def load(self, root):
        child_name = os.listdir(root)
        child_config_paths = []
        for child in child_name:
            child_path = os.path.join(root, child)
            child_config_paths.append(child_path)
        factory = Factory(child_config_paths)
        self.children = factory.produce()

