import torch
import os
class BaseModel:
    def __init__(self, num_samples: int, node_hash: int, epochs: int, batch_size: int, evaluating=False, device=None):
        self.num_samples = num_samples
        self.node_hash = node_hash
        self.epochs = epochs
        self.batch_size = batch_size
        self.evaluating = evaluating
        self.device = device if device else torch.device('cpu')
        self.data = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.model = None

        # get number of gpus
        cuda_devices = torch.cuda.device_count()
        self.device = 'cuda:' + str(self.node_hash % cuda_devices) if cuda_devices > 0 else 'cpu'

    def train(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError
    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)
        self.state_dict = self.model.state_dict()
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, weights_only=True))
    def save_model(self, path):
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        torch.save(self.model.state_dict(), path)