import torch
class BaseModel:
    def __init__(self, state_dict=None, *args, **kwargs):
        self.state_dict = state_dict

    def train(self):
        raise NotImplementedError
    def evaluate(self):
        raise NotImplementedError
    def init_weights(self, model):  
        if isinstance(model, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(model.weight)
            model.bias.data.fill_(0.01)
def get_model_by_name(name):
    if name == 'loan_defaulter':
        from .loan_defaulter import LoanDefaulter
        return LoanDefaulter