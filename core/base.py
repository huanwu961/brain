import os
import json
import time
import numpy as np
import utils
import taichi as ti

'''
-- Base Class --
Base class is a base data structure that describes an object with a bunch of large arrays and
a meta config file. It can automatically handle parameter saving, loading and monitoring.

It separates configuration files and array data, since array can be very large.

The base class only has configurations. Only when it is converted into actual sub-class, the
large arrays are initialized/loaded as taichi.field.

The config file has three parts:
-- meta: store those hyper-parameters
-- array: name, data type and shape of each arrays
-- log: monitor data, recording the macro indicators for each array can be dynamically 
        visualized in GUI
'''


class Base:
    def __init__(self,
                 name="root",           # name of the object
                 class_name="Base",     # sub-class name, for factory
                 log_frequency=10,      # number of normal updates between monitor data update
                 log_length=200,        # length of monitor log data
                 config=None            # other arguments
                 ):

        if config is None:
            # meta data
            self.name = name
            self.class_name = class_name
            self.time = time.time()

            # for future construction
            self.children = []

            self.configuration = {"meta": {}, "array": [], "log": []}

            self.base = True

            self.log_frequency = log_frequency
            self.log_length = log_length

            self.counter = 0

        # set member by config dictionary
        else:
            self.configuration = config
            for key, val in config['meta'].items():
                setattr(self, key, val)

    def save(self, root):
        # type check
        print("start saving...")
        if not isinstance(self, Base):
            print("children are not all class 'Base'")
            exit(0)

        # check root exists
        root_dir = os.path.join(root, self.name)
        os.makedirs(root_dir, exist_ok=True)
        print(f"saving path: {root_dir}")

        # save meta data into config.json
        print("saving meta data...")
        config_path = os.path.join(root_dir, 'config.json')
        array_dir = os.path.join(root_dir, 'array')
        self.config()
        self.configuration['array_path'] = array_dir
        json.dump(self.configuration, open(config_path, 'w'))

        # save array data into .npy
        print("saving arrays...")
        os.makedirs(array_dir, exist_ok=True)
        for _item_name in dir(self):
            _item = getattr(self, _item_name)
            if isinstance(_item, ti.Field):
                np.save(os.path.join(array_dir, _item_name + ".npy"), _item.to_numpy())

    def config(self):
        for item in dir(self):
            if item[0] != '_':
                if utils.json_check(getattr(self, item)):
                    # print("[json check]:", item, utils.json_check(getattr(self, item)))
                    self.configuration["meta"][item] = getattr(self, item)

        for _item_str in dir(self):
            _item = getattr(self, _item_str)
            if isinstance(_item, ti.Field):
                self.configuration["array"].append({
                    "name": _item_str,
                    "dtype": str(_item.dtype),
                    "shape": _item.shape
                })

    def monitor(self):
        log = {'time': time.time()}
        item_strs = dir(self)
        for item in item_strs:
            if isinstance(getattr(self, item), ti.VectorNdarray):
                array = getattr(self, item).to_numpy()

                array_log = {
                    "max": float(np.max(array)),
                    "min": float(np.min(array)),
                    "mean": float(np.mean(array)),
                    "std": float(np.std(array)),
                }
                log[item] = array_log
        self.configuration['log'].append(log)
        if len(self.configuration['log']) > self.log_length:
            self.configuration['log'].pop(0)