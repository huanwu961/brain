import os.path

from core import *
import json
import numpy as np
import taichi as ti

from area import NeuronArea
from area import SmallWorldArea
from connection import NeuronConnection
from sense import NeuronSense
from sense import VisualSense
from action import NeuronAction


class Factory:
    def __init__(self, config):
        self.configurations = config

    def init(self, config):
        class_name = config['meta']['class_name']
        if class_name == 'SmallWorldArea':
            return SmallWorldArea(config=config)
        if class_name == 'NeuronConnection':
            return NeuronConnection(config=config)
        if class_name == "VisualSense":
            return VisualSense(config=config)
        if class_name == "NeuronAction":
            return NeuronAction(config=config)
        if class_name == "NeuronArea":
            return NeuronArea(config=config)

    def produce(self):
        products = []
        for config in self.configurations:
            config = self.load_config(config)
            product = self.init(config)
            product = self.load_array(product)
            products.append(product)
        return products

    def load_config(self, config):
        # if the config is type str, it should be the location of the configuration file
        if isinstance(config, str):
            root = config
            if not os.path.isdir(config):
                print(f"[factory]: invalid directory '{root}', exiting..")
                exit(0)
            config_path = os.path.join(config, "config.json")
            try:
                config = json.load(open(config_path))
            except FileNotFoundError:
                print(f"[{config}]: config file not found")
                exit(0)

        elif not isinstance(config, dict):
            print("[factory]: unknown format")
            exit(0)

        if not self.is_config(config):
            print(f"[factory]: incorrect configuration format")
            exit(0)
        print(f"[factory]: {config}")
        return config

    def load_array(self, obj):
        for _item_str in dir(obj):
            _item = getattr(obj, _item_str)
            if isinstance(_item, ti.ScalarField):
                print(f'[{obj.name}]: loading array: ' + f'{_item_str}' + '...')
                try:
                    npy_path = os.path.join(obj.configuration['array_path'], _item_str + ".npy")
                    if os.path.exists(npy_path):
                        array = np.load(npy_path)
                        _item.from_numpy(array)
                    else:
                        print(f"[factory]: numpy file for '{_item_str}' not found")
                        exit()
                except FileNotFoundError:
                    print(f'[{_item_str}] not found, exit...')
                    exit()
        print('[%s]: finished loading array data' % obj.name)
        return obj

    def is_config(self, config):
        if isinstance(config, dict):
            if config.get('meta') is not None and config.get('array') is not None and config.get("log") is not None:
                return True
        else:
            return False

