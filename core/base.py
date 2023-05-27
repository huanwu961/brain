import os
import json
import time
import numpy as np


class Base:
    def __init__(self, name, base_type, **kwarg):
        # meta data
        self.name = name
        self.base_type = base_type
        self.time = time.time()

        # config
        self.config = kwarg
        self.config['name'] = name
        self.config['base_type'] = base_type

        # taichi field array
        self.array_list = []

        # for future construction
        self.children = []

    def save(self, root):
        # check root exists
        root_dir = os.path.join(root, self.name)
        os.makedirs(root_dir, exist_ok=True)

        # save meta data into config.json
        config_path = os.path.join(root_dir, 'config.json')
        json.dump(self.config, open(config_path, 'w'))

        # save array data into .npy
        array_dir = os.path.join(root_dir, 'array')
        os.makedirs(array_dir, exist_ok=True)
        for array in self.array_list:
            np.save(os.path.join(array_dir, array["name"]+".npy"), array['data'].to_numpy())

    def load_meta(self, root):
        try:
            self.config = json.load(open(os.path.join(root, 'config.json'), 'r'))
        except FileNotFoundError:
            print(f'[{root}] not found')
        print("[%s]: finish loading config file" % (self.config['name']))

    def load_array(self, root):
        for array in self.array_list:
            print("loading array: " + array['name'] + '...')
            try:
                array["data"].from_numpy(os.path.join(root, "array", array['name']+".npy"))
            except FileNotFoundError:
                print('[%s] not found, exit...' % (array["name"]))
                exit()
        print('[%s]: finished loading array data' % (self.config['name']))

    def load(self, root):
        self.load_meta(root)
        self.load_array(root)
