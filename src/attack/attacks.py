import torch
from logger import Logger
from skimage.util import random_noise
from collections import OrderedDict

class Noise:
    '''
    Modifed from: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/attacks/aggregation.py
        under GPL 3.0 License
    '''
    def __init__(self, attack_strength: float, logger: Logger):
        self.logger=logger
        self.attack_strength = attack_strength
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
    def __init__(self, strength=10000, logger: Logger=None):
        self.strength = strength
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
def create_attacker(attack_type, attack_strength, logger):
    if attack_type == 'noise_injection':
        return NoiseInjectionAttack(attack_strength, logger)
    elif attack_type == 'noise':
        return Noise(attack_strength, logger)
    else:
        raise ValueError(f'Unknown attack type: {attack_type}')