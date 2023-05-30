import os
import json
import time
import numpy as np
import utils
import taichi as ti


class Base:
    def __init__(self, name="root", class_name="Base", config=None, **kwarg):
        # meta data
        self.name = name
        self.class_name = class_name
        self.time = time.time()

        # for future construction
        self.children = []

        self.configuration = {"meta": {}, "array": [], "children": []}

    def save(self, root):
        # type check
        print("start saving...")
        if not utils.base_check(self, Base):
            print("children are not all class 'Base'")
            exit(0)

        # check root exists
        root_dir = os.path.join(root, self.name)
        os.makedirs(root_dir, exist_ok=True)
        print(f"saving path: {root_dir}")

        # save meta data into config.json
        print("saving meta data...")
        config_path = os.path.join(root_dir, 'config.json')
        json.dump(self.config(), open(config_path, 'w'))

        # save array data into .npy
        print("saving arrays...")
        array_dir = os.path.join(root_dir, 'array')
        os.makedirs(array_dir, exist_ok=True)
        for _item_name in dir(self):
            _item = getattr(self, _item_name)
            if isinstance(_item, ti.ScalarField):
                np.save(os.path.join(array_dir, _item_name + ".npy"), _item.to_numpy())

        # save child into children
        print("saving children...")
        children_dir = os.path.join(root_dir, 'children')
        os.makedirs(children_dir, exist_ok=True)
        for child in self.children:
            child.save(children_dir)

    def load(self, root):
        new_obj = self.load_config(root)
        self.children.append(new_obj)
        utils.recursive_cast(self)
        self.load_array(root, self)

    def load_config(self, config):
        if type(config) == str:
            try:
                config = json.load(open(config))
            except FileNotFoundError:
                print("[%s]: config file not found" % self.name)
                exit()
        obj = Base()
        for key, val in config["meta"].items():
            setattr(obj, key, val)

        for child_config in config["children"]:
            self.children.append(self.load_config(child_config))
        print("[%s]: finish loading config file" % self.name)
        return obj

    def load_array(self, root_path, root_obj):
        for _item_str in dir(root_obj):
            _item = getattr(root_obj, _item_str)
            if isinstance(_item, ti.ScalarField):
                print('loading array: ' + f'{_item_str}' + '...')
                try:
                    _item.from_numpy(os.path.join(root_path, "array", _item_str + ".npy"))
                except FileNotFoundError:
                    print(f'[{_item_str}] not found, exit...')
                    exit()
        print('[%s]: finished loading array data' % self.name)

        for child in root_obj.children:
            child_root_path = os.path.join(root_path, "children")
            self.load_array(child_root_path, child)

    def config(self):
        for item in dir(self):
            if item[0] != '_':
                if utils.json_check(getattr(self, item)):
                    self.configuration["meta"][item] = getattr(self, item)

        for _item_str in dir(self):
            _item = getattr(self, _item_str)
            if isinstance(_item, ti.ScalarField):
                self.configuration["array"].append({
                    "name": _item_str,
                    "dtype": str(_item.dtype),
                    "shape": _item.shape
                })

        for child in self.children:
            self.configuration["children"].append(child.config())

        return self.configuration


