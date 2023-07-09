from core.base import Base


class NeuronAction(Base):
    def __init__(self, name, config=None):
        super().__init__(name, 'NeuronAction', config=config)
        if config is None:
            pass

    def execute(self):
        pass
