import torch
from skimage.util import random_noise
from collections import OrderedDict
import copy
import math

device=None
class SignFlip:
    def __init__(self, attack_args: dict):
        # flipped model will go in opposite direction as the normally trained model
        self.flipped_model = None
    def attack(self, model: OrderedDict):
        print("[SignFlippingAttack]")
        for k in model:
            model[k] = -model[k]
        return model

class Noise():
    """
    Function to perform noise injection attack on the received weights.

    Modifed from: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/attacks/aggregation.py
        under GPL 3.0 License
    """
    def __init__(self, attack_args: dict):
        self.strength = attack_args['strength']


    def attack(self, model: OrderedDict):
        """
        Perform noise injection attack on the received weights. 
        :param received_weights: A dictionary containing the received weights.
        Returns A dictionary containing the noise injected weights.
        """
        print("[NoiseInjectionAttack] Performing noise injection attack")
        lkeys = list(model.keys())
        for k in lkeys:
            #print(f"Layer noised: {k}")
            # adding noise with mu =0, sigma=1 * strength
            model[k] = model[k].to(device)
            model[k].data += torch.randn(model[k].shape).to(device) * self.strength 
            
        return model

class RandomNoise():
    """
    Function to perform noise injection attack on the received weights.
    """
    def __init__(self, attack_args: dict):
        self.strength = attack_args['strength']
        print(f"[NoiseInjectionAttack] Initialized with strength: {self.strength}")

    def attack(self, model: OrderedDict):
        print("[NoiseInjectionAttack] Performing noise injection attack")
        random_model = copy.deepcopy(model)
        lkeys = list(random_model.keys())
        for k in lkeys:
            #print(f"Layer noised: {k}")
            # adding noise with mu =strength, sigma=1 * strength
            random_model[k] = torch.randn(random_model[k].shape).to(device) * self.strength
        return random_model

class ALittleIsEnough:
    def __init__(self, attack_args: dict):
        '''
        https://github.com/lishenghui/blades/blob/master/blades/adversaries/alie_adversary.py
        '''
        print(f"[ALIE] Initialized with args {attack_args}")
        num_nodes = attack_args['nodes']
        num_byz = attack_args['malicious_nodes']

        #s = torch.floor_divide(num_nodes, 2) + 1 - num_byz 
        # torch.floor_divide is deprecated. use torch.div 
        s = torch.div(num_nodes, 2, rounding_mode='trunc') + 1 - num_byz
        cdf_value = (num_nodes - num_byz - s) / (num_nodes - num_byz)
        dist = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0))
        self.z_max = dist.icdf(cdf_value)
    def attack(self, model_paths: list):
        '''
        ALIE attack on the model paths
        
        Args:
            model_paths: list of model paths
            
        Returns:
            poisoned_model: poisoned model
        '''
        # get mean of model and std of models
        poisoned_model = None
        std_model = None
        for model_path in model_paths:
            model = torch.load(model_path, map_location=device)
            if poisoned_model is None:
                poisoned_model = {name: torch.zeros_like(param).to(device) for name, param in model.items()}
                std_model = {name: torch.zeros_like(param).to(device) for name, param in model.items()}
            
            for layer in model:
                poisoned_model[layer] += model[layer].to(device)
                std_model[layer] += model[layer].to(device)**2

        for layer in poisoned_model:
            poisoned_model[layer] /= len(model_paths)
            std_model[layer] = torch.sqrt(std_model[layer]/len(model_paths) - poisoned_model[layer]**2)

            poisoned_model[layer] += std_model[layer] * self.z_max
        print("[ALIE] Poisoned model")
        return poisoned_model
      
# class InnerProductAttack:
#     """
#     Function to perform inner product attack on the received weights.
    
#     """
#     def __init__(self, attack_args: dict):
#         self.defense = attack_args['defense']
#         self.epsilon = attack_args['epsilon']
#         
#         print(f"[InnerProductAttack] Initialized for defense: {self.defense} with epsilon: {self.epsilon}")
#     def get_poisoned_model(self, models: list, prev_global: OrderedDict):
#         """
#         Get the mean of the models.
#         :param models: A list of tuples (model, num_samples) from benign clients.
#         :return: The mean model.
#         """
#         # Create a zero gradients
#         accum = {layer: torch.zeros_like(param) for layer, param in prev_global.items()}
        
#         for model in models:
#             for layer in accum:
#                 accum[layer] += (model[layer]-prev_global[layer])/len(models) # only add gradient vec
#         for layer in accum:
#             cop = torch.clone(accum[layer]).to('cpu').numpy()
#             print("Inner product" + str(cop.T.dot((cop*-1/len(models)))))
#             accum[layer] *= -1*self.epsilon
#             accum[layer]+=prev_global[layer]
#         return accum
#     def attack(self, models: list):
#         """
#         Perform inner product attack on the received weights.
#         :param models: A list of models where models[:-1] are benign and the last one is the previous global model
#         :return: A dictionary containing the attacked weights.
#         """
#         models, prev_global = models[:-1], models[-1]
#         print(f"[InnerProductAttack] Performing inner product attack num={len(models)}")
#         # Get the mean of the models
#         if self.defense=='fedavg':
#             attack_model = self.get_poisoned_model(models, prev_global)
#             return attack_model
        
#         else:
#             raise NotImplementedError(f"Defense {self.defense} not implemented")


    

def create_attacker(attack_type, attack_args, node_hash):
    global device
    num_gpus = torch.cuda.device_count()
    device = 'cuda:' + str(node_hash % num_gpus) if num_gpus > 0 else 'cpu'
    attack_type = attack_type.lower()
    if attack_type == 'noise':
        return Noise(attack_args)
    # elif attack_type == 'innerproduct':
    #     return InnerProductAttack(attack_args, logger)
    elif attack_type=='signflip':
        return SignFlip(attack_args)
    elif attack_type=='randomnoise':
        return RandomNoise(attack_args)
    elif attack_type =='alie':
        return ALittleIsEnough(attack_args)
    else:
        raise ValueError(f'Unknown attack type: {attack_type}')