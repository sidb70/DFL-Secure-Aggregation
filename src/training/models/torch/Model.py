import torch
class BaseModel:
    def __init__(self, num_samples: int, node_hash: int, epochs: int, batch_size: int, evaluating=False):
        self.num_samples = num_samples
        self.node_hash = node_hash
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluating = evaluating
        self.data = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.model = None

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.state_dict = self.model.state_dict()