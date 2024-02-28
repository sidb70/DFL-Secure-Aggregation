class Model:
    def __init__(self, state_dict=None):
        self.state_dict = state_dict

    def train(self):
        raise NotImplementedError
