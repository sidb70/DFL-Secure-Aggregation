import torch
from logger import Logger
from skimage.util import random_noise
from collections import OrderedDict
from aggregation.strategies import FedAvg

class Noise:
    '''
    Modifed from: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/attacks/aggregation.py
        under GPL 3.0 License
    '''
    def __init__(self, attack_args: dict, logger: Logger):
        self.logger=logger
        self.attack_strength = attack_args['strength']
    def attack(self, model: OrderedDict):
    # Function to add random noise of various types to the model parameter.
        poisoned_model = OrderedDict()

        for layer in model:
            bt = model[layer]
            t = bt.detach().clone()
            single_point = False
            if len(t.shape) == 0:
                t = t.view(-1)
                single_point = True
            # print(t)
            poisoned = torch.tensor(random_noise(t, mode='gaussian', mean=0, var=self.attack_strength, clip=True))
           
            if single_point:
                poisoned = poisoned[0]
            poisoned_model[layer] = poisoned

        return poisoned_model

class NoiseInjectionAttack():
    """
    Function to perform noise injection attack on the received weights.

    Modifed from: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/attacks/aggregation.py
        under GPL 3.0 License
    """
    def __init__(self, attack_args: dict, logger: Logger):
        self.strength = attack_args['strength']
        self.logger = logger

    def attack(self, received_weights):
        """
        Perform noise injection attack on the received weights. Takes a dictionary containing the received weights.
        Returns A dictionary containing the noise injected weights.
        """
        self.logger.log("[NoiseInjectionAttack] Performing noise injection attack")
        lkeys = list(received_weights.keys())
        for k in lkeys:
            self.logger.log(f"Layer noised: {k}")
            received_weights[k].data += torch.randn(received_weights[k].shape) * self.strength
        return received_weights
class InnerProductAttack:
    """
    Function to perform inner product attack on the received weights.
    
    """
    def __init__(self, attack_args: dict, logger: Logger):
        self.defense = attack_args['defense']
        self.epsilon = attack_args['epsilon']
        self.logger = logger
    def get_mean_model(self, models: list):
        """
        Get the mean of the models.
        :param models: A list of tuples (model, num_samples) from benign clients.
        :return: The mean model.
        """
        # Create a Zero Model
        accum = {layer: torch.zeros_like(param) for layer, param in models[-1].items()}
        
        for model in models:
            for layer, param in model.items():
                accum[layer] += param/len(models)
        return accum
    def attack(self, models: list):
        """
        Perform inner product attack on the received weights.
        :param models: A list of models from benign clients.
        :return: A dictionary containing the attacked weights.
        """
        self.logger.log("[InnerProductAttack] Performing inner product attack")
        # Get the mean of the models
        if self.defense=='mean':
            attack_model = self.get_mean_model(models)
            # multiply by negative epsilon
            for layer, param in attack_model.items():
                attack_model[layer] = (-1*self.epsilon)*param
            return attack_model
        
        else:
            raise NotImplementedError(f"Defense {self.defense} not implemented")


    

def create_attacker(attack_type, attack_args, logger):
    if attack_type == 'noise_injection':
        return NoiseInjectionAttack(attack_args, logger)
    elif attack_type == 'noise':
        return Noise(attack_args, logger)
    else:
        raise ValueError(f'Unknown attack type: {attack_type}')