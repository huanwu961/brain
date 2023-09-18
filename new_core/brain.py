import taichi as ti
import json
import time

class Brain:
    def __init__(self, name, root=None):
        self.name = name
        self.root = root

        self.groups = []
        self.connections = []
        self.senses = []
        self.actuators = []
        
    def run(self, duration=None, max_turn=None):
        start_time = time.time()

        # initialize
        self.init()

        # main loop for brain running
        turn = 0
        while True:
            for sensor in self.senses:
                sensor.read()
            for connection in self.connections:
                connection.update()
            for group in self.groups:
                group.update()
            for actuator in self.actuators:
                actuator.write()

        # stop and save training when duration or max_turn is reached
        turn += 1
        if turn >= max_turn:
            self.save(root=self.root)
            print(f"[Info]: brain {self.name} saved")
            print(f"[Info]: brain {self.name} finished [{turn} turns]")
            break
        elif time.time() - start_time >= duration:
            self.save(root=self.root)
            print(f"[Info]: brain {self.name} saved")
            print(f"[Info]: brain {self.name} finished [{time.time() - start_time} seconds]")
            break
        

    def init(self):
        # load groups
        for connection in self.connections:
            connection.init()

    # laod function
    def add_group(self, group):
        self.groups.append(group)

    def add_connection(self, connection):
        self.connections.append(connection)

    def add_sense(self, sense):
        self.senses.append(sense)

    def add_actuator(self, actuator):
        self.actuators.append(actuator)

    # get function
    def get_group(self, name):
        for group in self.groups:
            if group.name == name:
                return group
        print(f"[Error]: group {name} not found")
        return None

    def get_connection(self, name):
        for connection in self.connections:
            if connection.name == name:
                return connection
        print(f"[Error]: connection {name} not found")
        return None

    def get_sense(self, name):
        for sense in self.senses:
            if sense.name == name:
                return sense
        print(f"[Error]: sense {name} not found")
        return None

    def get_actuator(self, name):
        for actuator in self.actuators:
            if actuator.name == name:
                return actuator
        print(f"[Error]: actuator {name} not found")
        return None

        
