
import torch
import numpy as np
import copy
import yaml
import os

experiment_yaml = os.path.join('src','config', 'experiment.yaml')
with open(experiment_yaml) as f:
    experiment_params = yaml.safe_load(f)
device =None

def create_aggregator(node_hash):
    """
    Create an aggregator based on the aggregator type.

    Args:
        aggregator_type (str): The type of aggregator to create.

    Returns:
        Aggregator: An aggregator object.
    """
    global device
    global experiment_params
    aggregator_type = experiment_params['aggregation']
    num_gpus = torch.cuda.device_count()
    device = 'cuda:' + str(node_hash % num_gpus) if num_gpus > 0 else 'cpu'
    if aggregator_type == 'fedavg':
        return FedAvg()
    elif aggregator_type == 'median':
        return Median()
    elif aggregator_type == 'krum':
        return Krum()
    elif aggregator_type == 'trimmedmean':
        return TrimmedMean(beta=int(experiment_params['beta']))
    elif aggregator_type == 'geomed':
        return GeoMed()
    else:
        raise ValueError(f"Aggregator type {aggregator_type} not recognized")
class Aggregator:
    def __init__(self,  **kwargs):
        self.kwargs = kwargs
class FedAvg(Aggregator):
    """
    Source: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/learning/aggregators
    
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

    def aggregate(self, models_paths: list, log=True):
        """
        Weighted average of the models.

        Args:
            models: Dictionary with the models (node: model, num_samples).
        """
        if len(models_paths) == 0:
            print("[FedAvg] Trying to aggregate models when there are no models")
            return None


        # Total Samples
        # total_samples = sum(w for _, w in models_paths)

        # Create a Zero Model
        accum = None

        # Add weighted models
        if log:
            print(f"[FedAvg.aggregate] Aggregating models: num={len(models_paths)}")
        for model_path in models_paths:
            model = torch.load(model_path, map_location=device)
            if accum is None:
                accum = {layer: torch.zeros_like(param).to(device) for layer, param in model.items()}
            for layer in accum:
                accum[layer] += model[layer]
            del model
        # normalize by number of clients
        for layer in accum:
            accum[layer] /= len(models_paths)
        # Normalize by the number of samples
        # for layer in accum:
        #     accum[layer] /= total_samples
            
        # self.print_model_size(accum)

        return accum
class Median(Aggregator):
    """
    Source: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/learning/aggregators
    
    Median [Dong Yin et al., 2021]
    Paper: https://arxiv.org/pdf/1803.01498.pdf
    """

    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

    def get_median(self, weights):
        """
        Takes the median as the jth parameter
        of the global model. Note that when m is an even number,
        median is the mean of the middle two parameters.

        Args:
            weights: weights list, 2D tensor
        """

        # check if the weight tensor has enough space
        weight_len = len(weights)
        if weight_len == 0:
            print(
                "[Median] Trying to aggregate models when there is no models"
            )
            return None

            # get the median
        median = 0
        if weight_len % 2 == 1:
            # odd number, return the median
            median, _ = torch.median(weights, 0)
        else:
            # even number, return the mean of median two numbers
            # sort the tensor
            arr_weights = np.asarray(weights.cpu()) 
            nobs = arr_weights.shape[0]
            start = int(nobs / 2) - 1
            end = int(nobs / 2) + 1
            atmp = np.partition(arr_weights, (start, end - 1), 0)
            sl = [slice(None)] * atmp.ndim
            sl[0] = slice(start, end)
            arr_median = np.mean(atmp[tuple(sl)], axis=0)
            median = torch.tensor(arr_median,device=device)
        return median

    def aggregate(self, model_paths):
        """
        For each jth model parameter, the master device sorts the jth parameters of
        the m local models and takes the median as the jth parameter
        of the global model. Note that when m is an even number,
        median is the mean of the middle two parameters.

        Args:
            model_paths: List of model paths
        """
        # Check if there are models to aggregate
        if len(model_paths) == 0:
            print(
                               "[Median] Trying to aggregate models when there is no models"
            )
            return None

 

        total_models = len(model_paths)

        accum = None


        # Add weighteds models
        print("[Median.aggregate] Aggregating models: num={}".format(len(model_paths)))

        model = torch.load(model_paths[0], map_location=device)
        accum = {layer: torch.zeros_like(param).to(device) for layer, param in model.items()}
 

            
        for layer in accum:
            weight_layer = accum[layer].to(device)
            # get the shape of layer tensor
            l_shape = list(weight_layer.shape)

            # get the number of elements of layer tensor
            number_layer_weights = torch.numel(weight_layer)
            # if its 0-d tensor
            if l_shape == []:
                weights = torch.tensor([torch.load(model_paths[j])[layer] for j in range(0, total_models)])
                weights = weights.double()
                w = self.get_median(weights)
                accum[layer] = w

            else:
                # flatten the tensor
                weight_layer_flatten = weight_layer.view(number_layer_weights)

                # flatten the tensor of each model put all on the same device
                #models_layer_weight_flatten = torch.stack([models_params[j][layer].to(device).view(number_layer_weights) for j in range(0, total_models)], 0)
                models_layer_weight_flatten = torch.stack([torch.load(model_paths[j])[layer].to(device).view(number_layer_weights) for j in range(0, total_models)], 0)
                # get the weight list [w1j,w2j,··· ,wmj], where wij is the jth parameter of the ith local model
                median = self.get_median(models_layer_weight_flatten)
                accum[layer] = median.view(l_shape)


        return accum
class Krum(Aggregator):
    """
    Source: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/learning/aggregators
    
    Krum [Peva Blanchard et al., 2017]
    Paper: https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
    """

    def __init__(self,  **kwargs):
        super().__init__( **kwargs)

    def aggregate(self, model_paths):
        """
        Krum selects one of the m local models that is similar to other models
        as the global model, the euclidean distance between two local models is used.

        Args:
            model_paths: List of model paths
        """
        # Check if there are models to aggregate
        if len(model_paths) == 0:
            print(
                "[Krum] Trying to aggregate models when there is no models"
            )
            return None


        # Create a Zero Model
        model = torch.load(model_paths[0], map_location=device)
        accum = {layer: torch.zeros_like(param).to(device) for layer, param in model.items()}

        # Add weighteds models
        print("[Krum.aggregate] Aggregating models: num={}".format(len(model_paths)))

        # Create model distance list
        total_models = len(model_paths)
        distance_list = [0 for i in range(0, total_models)]

        # Calculate the L2 Norm between xi and xj
        min_index = 0
        min_distance_sum = float('inf')

        for i in range(0, total_models):
            m1 = torch.load(model_paths[i], map_location=device)
            for j in range(0, total_models):
                m2 = torch.load(model_paths[j], map_location=device)
                distance = 0
                if i == j:
                    distance = 0
                else:
                    for layer in m1:
                        l1 = m1[layer].to(device)
                        # l1 = l1.view(len(l1), 1)

                        l2 = m2[layer].to(device)
                        # l2 = l2.view(len(l2), 1)
                        
                        distance += torch.linalg.norm(l1 - l2)
                distance_list[i] += distance

            if min_distance_sum > distance_list[i]:
                min_distance_sum = distance_list[i]
                min_index = i

        
        # Assign the model with min distance with others as the aggregated model
        m = torch.load(model_paths[min_index], map_location=device)
        for layer in m:
            accum[layer]=accum[layer].to(device)
            accum[layer] = accum[layer] + m[layer].to(device)
            
        return accum


class TrimmedMean(Aggregator):
    """
    Source: https://github.com/lishenghui/blades/blob/master/blades/aggregators/aggregators.py

    TrimmedMean [Dong Yin et al., 2021]
    Paper: https://arxiv.org/pdf/1803.01498.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta = kwargs.get("beta", 0)
        self.num_byz = kwargs.get("byzantine", 0)
        self.num_excluded = int(self.beta * self.num_byz )
    def aggregate(self, model_paths):

        # Check if there are models to aggregate
        if len(model_paths) == 0:
            print(
                "[TrimmedMean] Trying to aggregate models when there is no models"
            )
            return None
        
        # Add weighteds models
        print("[TrimmedMean.aggregate] Aggregating models: num={}".format(len(model_paths)))
        num_models = len(model_paths)
        model = torch.load(model_paths[0], map_location=device)
        inputs = {layer: [] for layer in model}
        for model_path in model_paths:
            model = torch.load(model_path, map_location=device)
            for layer in model:
                inputs[layer].append(model[layer])
        for layer in inputs:
            # stack all the layers of each model
            inputs[layer] = torch.stack(inputs[layer], 0)

            # calculate the trimmed mean
            largest, _ = torch.topk(inputs[layer], self.num_excluded, 0)
            neg_smallest, _ = torch.topk(-inputs[layer], self.num_excluded, 0)
            new_stacked = torch.cat([inputs[layer], -largest, neg_smallest]).sum(0)
            new_stacked /= num_models - 2 * self.num_excluded
            inputs[layer] = new_stacked
        return inputs
class GeoMed:
    '''
    Source: https://github.com/lishenghui/blades/blob/master/blades/aggregators/aggregators.py
    '''
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.maxiter = kwargs.get("maxiter", 100)
        self.eps= kwargs.get("eps", 1e-6)
        self.ftol = kwargs.get("ftol", 1e-10)

    def aggregate(self, model_paths):
        # Check if there are models to aggregate
        if len(model_paths) == 0:
            print(
                "[GeoMed] Trying to aggregate models when there is no models"
            )
            return None

        # Add weighteds models
        print("[GeoMed.aggregate] Aggregating models: num={}".format(len(model_paths)))
        num_models = len(model_paths)
        model = torch.load(model_paths[0], map_location=device)
        inputs = {layer: [] for layer in model}
        for model_path in model_paths:
            model = torch.load(model_path, map_location=device)
            for layer in model:
                inputs[layer].append(model[layer])
        for layer in inputs:
            # stack all the layers of each model
            inputs[layer] = torch.stack(inputs[layer], 0)
            weights = torch.ones(num_models, device=device)
            def weighted_average(inps, w):
                # match size on singleton dimensions
                w = w.view(-1, *([1] * (inps.ndim - 1)))
                return torch.sum(inps * w, 0) / torch.sum(w)


            def obj_func(median, inp, weights):

                return np.average(
                    [torch.norm(p - median).item() for p in inp],
                    weights=weights.cpu(),
                )
            with torch.no_grad():
                median = weighted_average(inputs[layer], weights)
                new_weights = weights
                objective_value = obj_func(median, inputs[layer], weights)

                # Weiszfeld iterations
                for _ in range(self.maxiter):
                    prev_obj_value = objective_value
                    denom = torch.stack([torch.norm(p - median) for p in inputs[layer]])
                    new_weights = weights / torch.clamp(denom, min=self.eps)
                    median = weighted_average(inputs[layer], new_weights)

                    objective_value = obj_func(median, inputs[layer], weights)
                    if abs(prev_obj_value - objective_value) <= self.ftol * objective_value:
                        break

                median = weighted_average(inputs[layer], new_weights)
                inputs[layer] = median
        return inputs
