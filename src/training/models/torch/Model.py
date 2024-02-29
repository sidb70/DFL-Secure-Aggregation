import torch
class BaseModel:
    def __init__(self, data_file: str, num_samples: int, node_hash: int, epochs: int, batch_size: int,):
        self.data_file = data_file
        self.num_samples = num_samples
        self.node_hash = node_hash
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.model = None

    def train(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError
    def init_weights(self, model):  
        if isinstance(model, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.01)
