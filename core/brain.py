import random
import time
import psutil
import os
from base import Base
from area import NeuronArea
from action import NeuronAction
from sense import NeuronSense
from connection import NeuronConnection


class Brain(Base):
    def __init__(self, name="brain_"+str(random.randint(0, 1000000))):
        super().__init__(name, "Brain")
        self.areas = []
        self.connections = []
        self.senses = []
        self.actions = []
        self.info = {}
        self.base = False

    def init(self):
        for child in self.children:
            if isinstance(child, NeuronArea):
                self.areas.append(child)
            elif isinstance(child, NeuronSense):
                self.senses.append(child)
            elif isinstance(child, NeuronConnection):
                self.connections.append(child)
                self.areas.append(child.neuron_array)
            elif isinstance(child, NeuronAction):
                self.actions.append(child)

    def run(self):
        self.init()
        while True:
            self.info['start'] = time.time()
            for sensor in self.senses:
                sensor.read()
            self.info['read_time'] = time.time()
            for connection in self.connections:
                connection.connection_update()
                
            self.info['connection_time'] = time.time()
            for area in self.areas:
                area.update()
            self.info['main_update_time'] = time.time()
            for action in self.actions:
                action.act()
            self.print_info()

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

