import torch
from logger import Logger
class NoiseInjectionAttack():
    """
    Function to perform noise injection attack on the received weights.

    Modifed from: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/attacks/aggregation.py
        under GPL 3.0 License
    """
    def __init__(self, logger: Logger, strength=10000):
        self.strength = strength
        self.logger = logger

    def attack(self, received_weights):
        self.logger.log("[NoiseInjectionAttack] Performing noise injection attack")
        lkeys = list(received_weights.keys())
        for k in lkeys:
            self.logger.log(f"Layer noised: {k}")
            received_weights[k].data += torch.randn(received_weights[k].shape) * self.strength
        return received_weights
def create_attacker(attack_type, attack_strength, logger):
    if attack_type == 'noise':
        return NoiseInjectionAttack(logger, attack_strength)