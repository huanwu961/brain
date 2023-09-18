from base import Base
import taichi as ti
import json

class Connection(Base):
    def __init__(self,
                 from_name: str, # name of the pre-synaptic neuron group
                 to_name: str,  # name of the post-synaptic neuron group
                 connection_type: str='continuous', # continuous or discrete (default continuous)
                                               # if continous, then start and end positions are needed
                                               # if discrete, then the concrete connection matrix is needed
                 learnable: bool=False, # whether the connection is static or dynamic. If dynamic, then connection matrix is learnable
                 weight: float=1.0, # If static, then this is the default weight of the connection
                 from_start: tuple=(0, 0, 0, 0), # If static and continuous, then this is the start position of the out-edge
                 from_end: tuple=(0, 0, 0, 0), # If static and continuous, then this is the end position of the out-edge
                 to_start: tuple=(0, 0, 0, 0), # If static and continuous, then this is the start position of the in-edge
                 to_end: tuple=(0, 0, 0, 0), # If static and continuous, then this is the end position of the in-edge
                 protocol: dict=None, # This discribes the protocol of the data stored in Connection vector field
                ):
        # init base class
        Base.__init__(self, name=f'{from_name}->{to_name}')

        # init class members
        self.from_name = from_name
        self.to_name = to_name
        self.connection_type = connection_type
        self.update = update
        self.weight = weight
        self.from_start = ti.Vector(from_start)
        self.from_end = ti.Vector(from_end)
        self.to_start = ti.Vector(to_start)
        self.to_end = ti.Vector(to_end)
        self.protocol = protocol
        self.connection_matrix = ti.Vector.field(4, dtype=ti.f32, shape=(self.from_group.size(), self.to_group.size()))

    def initialize(self, brain):
        self.from_group = brain.get_group(self.from_name)
        self.to_group = brain.get_group(self.to_name)
        if self.from_group is None:
            raise Exception(f'Cannot find group {self.from_name}')
        if self.to_group is None:
            raise Exception(f'Cannot find group {self.to_name}')

        if self.connection_type == 'discrete':
            self.connection_matrix = None
    
    def update(self):
        if self.connection_type == 'continuous':
            self.update_continuous()
        if self.connection_type == 'discrete':
            self.update_discrete()
        
    @ti.kernel
    def update_continuous(self):
        from_range = self.from_end - self.from_start
        to_range = self.to_end - self.to_start
        ratio_vec = to_range / from_range

        # get the culmulative state and weight positions
        from_current_state_pos = self.from_group.protocol['current_state_pos']

        to_culmulative_state_pos = self.to_group.protocol['culmulative_state_pos']
        to_culmulative_weight_pos = self.to_group.protocol['culmulative_weight_pos']

        # main update process
        for I in ti.ndrange(from_range):
            from_pos = self.from_start + I

            for J in ti.ndrange(to_range):
                to_pos = ti.cast(self.to_start + J * ratio_vec, ti.i32)
                self.to_group.data[to_culmulative_state_pos] += self.from_group.data[from_culmulative_state_pos] * self.weight
                self.to_group.data[to_culmulative_weight_pos] += self.weight

    # TODO: implement discrete connection update
    @ti.kernel
    def update_discrete(self):
        pass
