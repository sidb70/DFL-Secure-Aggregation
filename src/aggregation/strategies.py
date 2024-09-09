import torch
import numpy as np
import yaml
import os

experiment_yaml = os.path.join('src','config', 'experiment.yaml')
with open(experiment_yaml) as f:
    experiment_params = yaml.safe_load(f)
device =None

def create_aggregator(node_hash, agg_args: dict = {}):
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
        return TrimmedMean(trimmed_mean_beta=agg_args.get("trimmed_mean_beta", 0.2))
    elif aggregator_type == 'geomed':
        return GeoMed()
    elif aggregator_type == 'multikrum':
        return MultiKrum(f=agg_args.get("f", None), m=agg_args.get("m", None))
    else:
        raise ValueError(f"Aggregator type {aggregator_type} not recognized")
class Aggregator:
    def __init__(self,  **kwargs):
        self.kwargs = kwargs
        self.device = kwargs.get("device", "cpu")
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
            model = torch.load(model_path, map_location=self.device)
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
            median = torch.tensor(arr_median,device=self.device)
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

        model = torch.load(model_paths[0], map_location=self.device)
        accum = {layer: torch.zeros_like(param).to(device) for layer, param in model.items()}
 

            
        for layer in accum:
            weight_layer = accum[layer].to(device)
            # get the shape of layer tensor
            l_shape = list(weight_layer.shape)

            # get the number of elements of layer tensor
            number_layer_weights = torch.numel(weight_layer)
            # if its 0-d tensor
            if l_shape == []:
                weights = torch.tensor([torch.load(model_paths[j], weights_only=True)[layer] for j in range(0, total_models)])
                weights = weights.double()
                w = self.get_median(weights)
                accum[layer] = w

            else:
                # flatten the tensor
                weight_layer_flatten = weight_layer.view(number_layer_weights)

                # flatten the tensor of each model put all on the same device
                #models_layer_weight_flatten = torch.stack([models_params[j][layer].to(device).view(number_layer_weights) for j in range(0, total_models)], 0)
                models_layer_weight_flatten = torch.stack([torch.load(model_paths[j], weights_only=True)[layer].to(device).view(number_layer_weights) for j in range(0, total_models)], 0)
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
        model = torch.load(model_paths[0], map_location=self.device,weights_only=True)
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
            m1 = torch.load(model_paths[i], map_location=self.device,weights_only=True)
            for j in range(0, total_models):
                m2 = torch.load(model_paths[j], map_location=self.device,weights_only=True)
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
        m = torch.load(model_paths[min_index], map_location=self.device,weights_only=True)
        for layer in m:
            accum[layer]=accum[layer].to(device)
            accum[layer] = accum[layer] + m[layer].to(device)
            
        return accum
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
        model = torch.load(model_paths[0], map_location=self.device,weights_only=True)
        inputs = {layer: [] for layer in model}
        for model_path in model_paths:
            model = torch.load(model_path, map_location=self.device,weights_only=True)
            for layer in model:
                inputs[layer].append(model[layer])
        for layer in inputs:
            # stack all the layers of each model
            inputs[layer] = torch.stack(inputs[layer], 0)
            weights = torch.ones(num_models, device=self.device)
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



def torch_full_partition(tensor, start, end):
     #print("Sorting", tensor.shape, "start", start, "end", end)
    sorted_tensor, _ = torch.sort(tensor, dim=0)
    # print("Done sorting")
    return sorted_tensor[start:end]
class TrimmedMean(Aggregator):
    """
    TrimmedMean [Dong Yin et al., 2021]
    Paper: https://arxiv.org/pdf/1803.01498.pdf
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.beta = kwargs.get("trimmed_mean_beta", 0.2)

    def get_trimmedmean(self, weights):
        """
        For weight list [w1j,w2j,··· ,wmj], removes the largest and
        smallest β of them, and computes the mean of the remaining
        m-2β parameters

        Args:
            weights: weights list, 2D tensor
        """

        # check if the weight tensor has enough space
        weight_len = len(weights)
        if weight_len == 0:
            print(
                "[TrimmedMean] Trying to aggregate models when there is no models"
            )
            return None
        if weight_len <= 2 * self.beta * weight_len:
            # logging.error(
            #     "[TrimmedMean] Number of model should larger than 2 * beta"
            # )
            remaining_wrights = weights
            res = torch.mean(remaining_wrights, 0)

        else:
            num_trim = int(2 * self.beta * weight_len)
            nobs = weights.shape[0]
            start = int(num_trim / 2)
            end = nobs - int(num_trim / 2)
            # # remove the largest and smallest β items
            # arr_weights = np.asarray(weights.cpu())
           
            # atmp = np.partition(arr_weights, (start, end - 1), 0)

            # # Use torch.kthvalue to find the start-th and end-th elements
            # start_val = torch.kthvalue(weights, start + 1).values
            # end_val = torch.kthvalue(weights, end).values
            # # Create a mask for values between start_val and end_val
            # mask = (weights >= start_val) & (weights <= end_val)
            # # Use the mask to get the "partitioned" tensor
            # atmp = weights[mask]

            # sl = [slice(None)] * atmp.ndim
            # sl[0] = slice(start, end)

            # # # print(atmp[tuple(sl)])
            # # arr_median = np.mean(atmp[tuple(sl)], axis=0)
            # arr_median = torch.mean(atmp, 0)

            # print("Starting trim")
            trimmed_weights = torch_full_partition(weights, start, end)
            # print("Done with trim")
            res = torch.mean(trimmed_weights, 0)
            # res = torch.tensor(arr_median)


        # get the mean of the remaining weights
        return res

    def aggregate(self, model_paths):
        """
        For each jth model parameter, the master device sorts the jth parameters
        of the m local models, i.e., w1j,w2j,··· ,wmj, where wij is the
        jth parameter of the ith local model, removes the largest and
        smallest β of them, and computes the mean of the remaining
        m-2β parameters as the jth parameter of the global model.

        Args:
            model_paths: List of model paths
        """
        # Check if there are models to aggregate
        if len(model_paths) == 0:
            print(
                "[TrimmedMean] Trying to aggregate models when there is no models"
            )
            return None
        total_models = len(model_paths)
        # models = list(models.values())
        # models_params = [m for m, _ in models]

        # Create a Zero Model
        # accum = (models[-1][0]).copy()
        self.device = 'cuda:0'
        model = torch.load(model_paths[0],weights_only=True, map_location=self.device)
        accum = {layer: torch.zeros_like(param).to(self.device) for layer, param in model.items()}

        # Add weighted models
        print("[TrimmedMean.aggregate] Aggregating models: num={}".format(len(model_paths)))

        model_weights = [torch.load(model_path, weights_only=True, map_location=self.device) for model_path in model_paths]

        # Calculate the trimmedmean for each parameter
        for layer in accum:
            weight_layer = accum[layer]
            # get the shape of layer tensor
            l_shape = list(weight_layer.shape)

            # get the number of elements of layer tensor
            number_layer_weights = torch.numel(weight_layer)
            # if its 0-d tensor
            if l_shape == []:
                weights = torch.tensor([model_weights[j][layer] for j in range(0, total_models)]).to(self.device)
                weights = weights.double()
                w = self.get_trimmedmean(weights)
                accum[layer] = w

            else:
                # flatten the tensor
                weight_layer_flatten = weight_layer.view(number_layer_weights)

                # flatten the tensor of each model
                models_layer_weight_flatten = torch.stack([model_weights[j][layer].view(number_layer_weights) for j in range(0, total_models)], 0)
                # print("Weights shape", models_layer_weight_flatten.shape)
                
                # get the weight list [w1j,w2j,··· ,wmj], where wij is the jth parameter of the ith local model
                trimmedmean = self.get_trimmedmean(models_layer_weight_flatten)
                accum[layer] = trimmedmean.view(l_shape)

        return accum

def _compute_scores(distances, i, n, f):
    """Compute scores for node i.

    Args:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
        i, j starts with 0.
        i {int} -- index of worker, starting from 0.
        n {int} -- total number of workers
        f {int} -- Total number of Byzantine workers.

    Returns:
        float -- krum distance score of i.
    """
    s = [distances[j][i] ** 2 for j in range(i)] + [
        distances[i][j] ** 2 for j in range(i + 1, n)
    ]
    _s = sorted(s)[: n - f - 2]
    return sum(_s)


def _multi_krum(distances, n, f, m):
    """Multi_Krum algorithm.

    Arguments:
        distances {dict} -- A dict of dict of distance. distances[i][j] = dist.
         i, j starts with 0.
        n {int} -- Total number of workers.
        f {int} -- Total number of Byzantine workers.
        m {int} -- Number of workers for aggregation.

    Returns:
        list -- A list indices of worker indices for aggregation. length <= m
    """
    if n < 1:
        raise ValueError(
            "Number of workers should be positive integer. Got {}.".format(f)
        )

    if m < 1 or m > n:
        raise ValueError(
            "Number of workers for aggregation should be >=1 and <= {}. Got {}.".format(
                m, n
            )
        )

    if 2 * f + 2 > n:
        raise ValueError("Too many Byzantine workers: 2 * {} + 2 >= {}.".format(f, n))

    for i in range(n - 1):
        for j in range(i + 1, n):
            if distances[i][j] < 0:
                raise ValueError(
                    "The distance between node {} and {} should be non-negative: "
                    "Got {}.".format(i, j, distances[i][j])
                )

    scores = [(i, _compute_scores(distances, i, n, f)) for i in range(n)]
    sorted_scores = sorted(scores, key=lambda x: x[1])
    return list(map(lambda x: x[0], sorted_scores))[:m]


def _compute_euclidean_distance(v1, v2):
    return (v1 - v2).norm()


def _pairwise_euclidean_distances(vectors):
    """Compute the pairwise euclidean distance.

    Arguments:
        vectors {list} -- A list of vectors.

    Returns:
        dict -- A dict of dict of distances {i:{j:distance}}
    """
    n = len(vectors)
    vectors = [v.flatten() for v in vectors]

    distances = {}
    for i in range(n - 1):
        distances[i] = {}
        for j in range(i + 1, n):
            distances[i][j] = _compute_euclidean_distance(vectors[i], vectors[j]) ** 2
    return distances
class MultiKrum:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.f = kwargs.get("f", 1)
        self.m = kwargs.get("m", 1)
    def aggregate(self, model_paths: list):
        if len(model_paths) == 0:
            print("[MultiKrum] Trying to aggregate models when there are no models")
            return None
        print("[MultiKrum.aggregate] Aggregating models: num={}".format(len(model_paths)))

        num_models = len(model_paths)
        model = torch.load(model_paths[0], map_location=self.device)
        inputs = {layer: [] for layer in model}
        for model_path in model_paths:
            model = torch.load(model_path, map_location=self.device,weights_only=True)
            for layer in model:
                inputs[layer].append(model[layer])
        for layer in inputs:
            # stack all the layers of each model
            updates = torch.stack(inputs[layer], 0)
            distances = _pairwise_euclidean_distances(updates)
            top_m_indices = _multi_krum(distances, len(updates), self.f, self.m)
            values = torch.stack([updates[i] for i in top_m_indices], dim=0).mean(dim=0)
            inputs[layer] = values
        return inputs
