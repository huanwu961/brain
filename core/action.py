from base import Base


class NeuronAction(Base):
    def __init__(self, name):
        super().__init__(name, 'NeuronAction')
        pass
    
    def execute(self):
        pass
