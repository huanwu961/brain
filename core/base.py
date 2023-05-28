import os
import json
import time
import numpy as np
import utils
import taichi as ti


class Base:
    def __init__(self, name="default", base_type="base", **kwarg):
        # meta data
        self.name = name
        self.base_type = base_type
        self.time = time.time()

        # taichi field array
        self.array_list = {}

        # for future construction
        self.children = []

    def save(self, root):
        # type check
        if not utils.base_check(self, Base):
            print("children are not all class 'Base'")
            exit(0)

        # check root exists
        root_dir = os.path.join(root, self.name)
        os.makedirs(root_dir, exist_ok=True)

        # save meta data into config.json
        config_path = os.path.join(root_dir, 'config.json')
        json.dump(self.config(), open(config_path, 'w'))

        # save array data into .npy
        array_dir = os.path.join(root_dir, 'array')
        os.makedirs(array_dir, exist_ok=True)
        for name, array in self.array_list:
            np.save(os.path.join(array_dir, name + ".npy"), array.to_numpy())

        # save child into children
        children_dir = os.path.join(root_dir, 'children')
        os.makedirs(children_dir, exist_ok=True)
        for child in self.children:
            child.save(children_dir)

    def load(self, root):
        try:
            config = json.load(open(os.path.join(root, 'config.json'), 'r'))
            self.new(config)
        except FileNotFoundError:
            print(f'[{root}] not found')
        print("[%s]: finish loading config file" % self.name)

        for name, array in self.array_list:
            print('loading array: ' + f'{name}' + '...')
            try:
                array.from_numpy(os.path.join(root, "array", array['name'] + ".npy"))
            except FileNotFoundError:
                print(f'[{name}] not found, exit...')
                exit()
        print('[%s]: finished loading array data' % self.name)

        for child in self.children:
            child_root = os.path.join(root, "children")
            child.load(child_root)

    def new(self, config):
        for key, val in config["meta"]:
            setattr(self, key, val)

        for array_config in config["array"]:
            dtype = 'ti.' + array_config['dtype']
            shape = array_config['shape']
            name = array_config['name']
            cmd = f"setattr(self, {name}, ti.field(dtype={dtype}, shape={shape})"
            exec(cmd)

        for child_config in config["children"]:
            obj = Base()
            self.children.append(obj.new(child_config))

    def config(self):
        config = {"meta": {}, "array": [], "children": []}
        for item in dir(self):
            if item[0] is not '_':
                if utils.json_check(getattr(self, item)):
                    config["meta"][item] = getattr(self, item)

        for name, array in self.array_list:
            config["array"].append({
                "name": name,
                "dtype": str(array.dtype),
                "shape": array.shape
            })

        for child in self.children:
            config["children"].append(child.config())

        return config


