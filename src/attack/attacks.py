import torch
from logger import Logger
from skimage.util import random_noise
from collections import OrderedDict

class NoiseInjectionAttack():
    """
    Function to perform noise injection attack on the received weights.

    Modifed from: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/attacks/aggregation.py
        under GPL 3.0 License
    """
    def __init__(self, attack_args: dict, logger: Logger):
        self.strength = attack_args['strength']
        self.logger = logger

    def attack(self, model: OrderedDict):
        """
        Perform noise injection attack on the received weights. 
        :param received_weights: A dictionary containing the received weights.
        Returns A dictionary containing the noise injected weights.
        """
        self.logger.log("[NoiseInjectionAttack] Performing noise injection attack")
        lkeys = list(model.keys())
        for k in lkeys:
            self.logger.log(f"Layer noised: {k}")
            model[k].data += torch.randn(model[k].shape) * self.strength
        return model
class InnerProductAttack:
    """
    Function to perform inner product attack on the received weights.
    
    """
    def __init__(self, attack_args: dict, logger: Logger):
        self.defense = attack_args['defense']
        self.epsilon = attack_args['epsilon']
        self.logger = logger
        self.logger.log(f"[InnerProductAttack] Initialized for defense: {self.defense} with epsilon: {self.epsilon}")
    def get_poisoned_model(self, models: list):
        """
        Get the mean of the models.
        :param models: A list of tuples (model, num_samples) from benign clients.
        :return: The mean model.
        """
        # Create a Zero Model
        accum = {layer: torch.zeros_like(param) for layer, param in models[-1].items()}
        
        for model in models:
            for layer, param in model.items():
                accum[layer] += param 
        for layer, param in accum.items():
            accum[layer] *= -1*self.epsilon/len(models)
        return accum
    def attack(self, models: list):
        """
        Perform inner product attack on the received weights.
        :param models: A list of models from benign clients.
        :return: A dictionary containing the attacked weights.
        """
        self.logger.log(f"[InnerProductAttack] Performing inner product attack num={len(models)}")
        # Get the mean of the models
        if self.defense=='fedavg':
            attack_model = self.get_poisoned_model(models)
            return attack_model
        
        else:
            raise NotImplementedError(f"Defense {self.defense} not implemented")


    

def create_attacker(attack_type, attack_args, logger):
    if attack_type == 'noiseinjection':
        return NoiseInjectionAttack(attack_args, logger)
    elif attack_type == 'innerproduct':
        return InnerProductAttack(attack_args, logger)
    else:
        raise ValueError(f'Unknown attack type: {attack_type}')