import json
import numpy as np
import os

from brain import Brain
from area import NueronArea
from connection import NueronConnection
from sense import VisualSense
from action import NueronAction
import time

import threading

class Data:
    def __init__(self, root_path, backup_path=None,auto_save=False):
        auto_save_thread = threading.Thread(target=self.auto_save)
        self.objects = []
        self.last = time.time()
        self.now = time.time()
        self.auto_save_interval = 60
        self.auto_save = auto_save
        self.backup_path = backup_path
        self.root_path = root_path
        
        if os.path.exists(root_path) == False:
            os.mkdir(root_path)
        
    
    def add(self, obj):
        self.objects.append(obj)
    
    def save(self, obj, path=""):
        if path == "":
            path = self.root_path
        if obj.type == "brain":
            self.save_brain(obj, path)
        elif obj.type == "nueron_array":
            self.save_nueron_area(obj, path)
        elif obj.type == "nueron_connection":
            self.save_nueron_connection(obj, path)
        elif obj.type == "nueron_action":
            self.save_nueron_action(obj, path)
        elif obj.type == "nueron_sense":
            self.save_nueron_sense(obj, path)
        else:
            print("Save error: unknown type '%s'" % obj.type)
            
            
    def load(self, path=None, name=None):
        if path == None:
            path = self.path
            if name == None:
                print("Load error: no path or name given")
                return
            else:
                path = os.path.join(path, name)
                
        try:
            data = json.load(open(os.path.join(path, "config.json"), "r"))
        except:
            print("Load error: config file not found")
            return
        
        type = data['type']
        if type == "brain":
            obj = self.load_brain(path)
        elif type == "nueron_array":
            obj = self.load_nueron_array(path)
        elif type == "nueron_connection":
            obj = self.load_nueron_connection(path)
        elif type == "nueron_action":
            obj = self.load_nueron_action(path)
        elif type == "nueron_sense":
            obj = self.load_nueron_sense(path)
        else:
            print("Load error: unknown type '%s'" % type)
            
        self.add(obj)
        return obj
            

    def auto_save(self, obj, disk_path=""):
        disks = os.listdir("/Volumes")
        disks.remove("Macintosh HD")
        if os.path.exists(disk_path):
            print("Auto save error: diretory '%s' not found" % disk_path)
            return
        elif disk_path == "":
            backup_path = os.path.join("/Volumes", disks[0], "BrainBackup")
        else:
            backup_path = os.path.join(disk_path, "BrainBackup")
        
        config = {}
        config['name'] = obj.name
        config['type'] = obj.type
        config['time'] = time.time()
        
        if not os.path.exists(backup_path):
            os.mkdir(backup_path)
            configs = []
            configs.append(config)
            json.dump(configs, open(os.path.join(backup_path, "config.json"), "w"))
        else:
            configs = json.load(open(os.path.join(backup_path, "config.json"), "r"))
            configs.append(config)
            json.dump(configs, open(os.path.join(backup_path, "config.json"), "w"))
        
            
        while True:
            self.now = time.time()
            if self.now - self.last > self.auto_save_interval:
                self.save(obj, obj.path)
                self.last = self.now
            
        
    
    def save_nueron_area(self, narray, path):
        # define root path
        root = os.path.join(path, narray.name)
        
        # create directory
        os.makedirs(root, exist_ok=True)
        
        # save config
        config = {}
        config['name'] = narray.name
        config['type'] = narray.type
        config['n'] = narray.n
        config['m'] = narray.m
        config['topology'] = narray.topology
        
        json.dump(config, open(os.path.join(root, 'config.json'), 'w'))
        
        # create data directory
        data_root = os.path.join(root, 'data')
        os.makedirs(data_root, exist_ok=True)
        
        # save data
        np.save(os.path.join(data_root, 'current_state.npy'), narray.current_state.to_numpy())
        np.save(os.path.join(data_root, 'last_state.npy'), narray.last_state.to_numpy())
        np.save(os.path.join(data_root, 'cumulative_state.npy'), narray.cumulative_state.to_numpy())
        np.save(os.path.join(data_root, 'cumulative_weight.npy'), narray.cumulative_weight.to_numpy())
        np.save(os.path.join(data_root, 'weights.npy'), narray.weight.to_numpy())
        
    def load_nueron_area(self, path):
        # load config
        config = json.load(open(os.path.join(path, 'config.json'), 'r'))
        
        # create nueron array
        narray = NueronArea(config['n'], config['m'], config['name'])
        
        # load data
        data_root = os.path.join(path, 'data')
        narray.current_state.from_numpy(np.load(os.path.join(data_root, 'current_state.npy')))
        narray.last_state.from_numpy(np.load(os.path.join(data_root, 'last_state.npy')))
        narray.cumulative_state.from_numpy(np.load(os.path.join(data_root, 'cumulative_state.npy')))
        narray.cumulative_weight.from_numpy(np.load(os.path.join(data_root, 'cumulative_weight.npy')))
        narray.weight.from_numpy(np.load(os.path.join(data_root, 'weights.npy')))
        
        return narray
        
    def save_nueron_connection(self, nconnection, path):
        # define root path
        root = os.path.join(path, nconnection.name)
        
        # create directory
        os.makedirs(root, exist_ok=True)
        
        # save config
        config = {}
        config['name'] = nconnection.name
        config['type'] = nconnection.type
        config['in_name'] = nconnection.in_array.name
        config['out_name'] = nconnection.out_array.name
        config['in_pos'] = nconnection.in_pos
        config['out_pos'] = nconnection.out_pos
        config['weight'] = nconnection.weight
        config['m'] = nconnection.m
        config['in_length'] = nconnection.in_length
        config['out_length'] = nconnection.out_length
        
        json.dump(config, open(os.path.join(root, 'config.json'), 'w'))
        
        # create data directory
        data_root = os.path.join(root, 'data')
        os.makedirs(data_root, exist_ok=True)
        
        # save current state
        np.save(os.path.join(data_root, 'output_position.npy'), nconnection.output_position.to_numpy())
        
    def load_nueron_connection(self, path):
        # load config
        config = json.load(open(os.path.join(path, 'config.json'), 'r'))
        
        # create nueron connection
        print(config['in_pos'], config['out_pos'], config['weight'], config['m'])
        nconnection = NueronConnection(None, None, config['in_pos'], config['out_pos'], config['weight'], config['m'])
        nconnection.in_name = config['in_name']
        nconnection.out_name = config['out_name']
        
        # load data
        data_root = os.path.join(path, 'data')
        nconnection.output_position.from_numpy(np.load(os.path.join(data_root, 'output_position.npy')))
        
        return nconnection
    
    def save_nueron_sense(self, nsense, path):
        # define root path
        root = os.path.join(path, nsense.name)
        
        # create directory
        os.makedirs(root, exist_ok=True)
        
        # save config
        config = {}
        config['name'] = nsense.name
        config['type'] = nsense.type
        config['source_type'] = nsense.source_type
        config['shape'] = nsense.shape
        config['source'] = nsense.source
        config['dim'] = nsense.dim
        config['size'] = nsense.size
        
        json.dump(config, open(os.path.join(root, 'config.json'), 'w'))
        
    def load_nueron_sense(self, path):
        # load config
        config = json.load(open(os.path.join(path, 'config.json'), 'r'))
        
        # create nueron sense
        if config['source_type'] == 'visual':
            print(config['shape'])
            nsense = VisualSense(config['source'], config['shape'], config['name'])
        else:
            print("not implemented yet")
        
        return nsense
    
    def save_brain(self, brain, path):
        # define root path
        root = os.path.join(path, brain.name)
        
        # create directory
        os.makedirs(root, exist_ok=True)

        # save config
        data = {}
        data['name'] = brain.name
        data['type'] = brain.type
        data['areas'] = [area.name for area in brain.areas]
        data['connections'] = [connection.name for connection in brain.connections]
        data['senses'] = [sense.name for sense in brain.senses]
        data['actions'] = [action.name for action in brain.actions]
        
        json.dump(data, open(os.path.join(root, 'config.json'), 'w'))
        
        # create data directory
        os.makedirs(os.path.join(root, "areas"), exist_ok=True)
        os.makedirs(os.path.join(root, "connections"), exist_ok=True)
        os.makedirs(os.path.join(root, "senses"), exist_ok=True)
        os.makedirs(os.path.join(root, "actions"), exist_ok=True)
        
        # save sub objects
        for area in brain.areas:
            area_path = os.path.join(root, "areas")
            print(area_path)
            self.save_nueron_area( area, path=area_path)
        for connection in brain.connections:
            self.save_nueron_connection(connection, os.path.join(root, "connections"))
        for sense in brain.senses:
            self.save_nueron_sense(sense, os.path.join(root, "senses"))
        for action in brain.actions:
            self.save_nueron_action(action, os.path.join(root, "actions"))
            
    def load_brain(self, path):
        # load config
        config = json.load(open(os.path.join(path, 'config.json'), 'r'))
        
        # create brain
        brain = Brain(config['name'])
        
        # load sub objects
        for area_name in config['areas']:
            brain.areas.append(self.load_nueron_area(os.path.join(path, "areas", area_name)))
        for connection_name in config['connections']:
            conn = self.load_nueron_connection(os.path.join(path, "connections", connection_name))
            print(conn.in_name, conn.out_name)
            in_array = brain.get_nueron_area(conn.in_name)
            out_array = brain.get_nueron_area(conn.out_name)
            conn.connect(in_array, out_array)
            brain.connections.append(conn)
        for sense_name in config['senses']:
            sense_area = self.load_nueron_sense(os.path.join(path, "senses", sense_name))
            sense_area.connect(brain.get_nueron_area(sense_area.name))
            brain.senses.append(sense_area)
            
        for action_name in config['actions']:
            brain.actions.append(self.load_nueron_action(os.path.join(path, "actions", action_name)))
            
        return brain
        
    def save_nueron_action(self, naction, path):
        pass
    
    def load_nueron_action(self, path):
        pass
        
        
        