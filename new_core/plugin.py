import importlib
import json

'''
This is the plugin system for adding new features to the framework. It will
be useful when we are trying different network topologies and different training
rules.
'''

class Plugin:
    def __init__(self) -> None:
        self.source_list = [] # store the source path of the plugins
        self.classes = {} # store the class name and class constructor
        self.config = {} # store the loaded config.json file
        
    # execute init() within the plugin file, which will register the class to self.classes
    def load_plugin(self, path):
        try:
            module = importlib.import_module(path)
        except ModuleNotFoundError:
            print("[Error]: module %s not found" % path)
            return
        module.init()
           
    # load all plugins in the source list
    def load_plugins(self):
        for path in self.source_list:
            self.load_plugin(path)
    
    # add the source path of new algorithms to the source list, and save it to config.json
    def module_register(self, source_path):
        self.source_list = json.load(open("../config/config.json"))
        if source_path not in self.source_list:
            self.source_list.append(source_path)
        json.dump(self.source_list, open("../config/config.json", "w"))
        
    # register the class to self.classes
    def class_register(self, class_name, cls):
        if class_name in self.classes:
            print("[Warning]: class %s already exists, will be overwritten" % class_name)
        self.classes[class_name] = cls
        
    # method for creating new instance of a class
    def create(self, class_name, *args, **kwargs):
        cls = self.classes.get(class_name)
        if not cls:
            raise ValueError(class_name)
        return cls(*args, **kwargs)