import json
import random
import time
import cv2
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
    def __init__(self, name="brain_"+str(random.randint(0, 1000000)), root='.', config=None):
        self.name = name
        self.areas = []
        self.connections = []
        self.senses = []
        self.actions = []
        self.children = []
        self.info = {}
        self.root = root
        if config is not None:
            self.config = config
        else:
            self.config ={"name": self.name, "class_name": "Brain"}

    def prepare(self):
        # load areas first, preventing void connection
        for child in self.children:
            if isinstance(child, NeuronArea):
                self.areas.append(child)
                print("[prepare area]:", child.name, child.n, child.current_state.shape)

        # append children to the correct area
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
                child.connect(self.get_child(area_1_name), self.get_child(area_2_name))

    def run(self, duration=None, max_turn=None, auto_save=False, save_gap=100):
        start_time = time.time()
        self.prepare()
        turn = 0
        while True:
            turn_start = time.time()
            self.info['start'] = time.time()
            for sensor in self.senses:
                sensor.read()
            self.info['read_time'] = time.time()
            for connection in self.connections:
                # print("connection:", connection.in_array.current_state.to_numpy())
                connection.update()

            self.info['connection_time'] = time.time()
            for area in self.areas:
                area.update()
                # print("area:", area.current_state.to_numpy())
            self.info['main_update_time'] = time.time()
            for action in self.actions:
                action.act()
            for child in self.children:
                # print(f"[{child.name}]: logging...")
                child.monitor()
            for area in self.areas:
                area.clear_cumulative()
            turn_end = time.time()
            if turn % 10 == 0:
                print(f"[training]: {turn} turn, ({1/(turn_end - turn_start)} fps)")
            if turn % save_gap == 0:
                if auto_save is True:
                    self.save(self.root)




            turn += 1
            if turn >= max_turn:
                # self.save(root=self.root)
                print(f"training finished, [{turn} turns]")
                break
            elif time.time() - start_time > duration:
                print(f"training finished, [{duration}s]")
                break



    def add(self, obj):
        self.children.append(obj)

    def save(self, root):
        # save children objects
        brain_root = os.path.join(root, self.name)
        os.makedirs(brain_root, exist_ok=True)

        for child in self.children:
            print('[configuration]: ', child.configuration)
            child.save(brain_root)

        # save config file
        config_path = os.path.join(brain_root, "config.json")
        with open(config_path, 'w+') as f:
            json.dump(self.config, f)

    def load(self, root):
        # load child object
        child_name = os.listdir(root)
        child_config_paths = []
        for child in child_name:
            child_path = os.path.join(root, child)
            if os.path.isdir(child_path):
                child_config_paths.append(child_path)
        factory = Factory(child_config_paths)
        self.children = factory.produce()

        # load config file
        config_path = os.path.join(root, "config.json")
        with open(config_path, 'r+') as f:
            self.config = json.load(f)

    def search(self, root):
        brains = []
        os.chdir(root)
        names = os.listdir()
        for name in names:
            config_path = os.path.join(os.getcwd(), name, "config.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    config = json.load(f)
                if config["class_name"] is not None and config["class_name"] == "Brain":
                    brains.append(os.path.join(os.getcwd(), name))
        return brains

    # ----------------- get neuron object -----------------
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

    def get_child(self, name):
        for child in self.children:
            if child.name == name:
                return child
        print(f"[brain]: child {name} not found!")
        return None