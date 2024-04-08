'''
Source: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/learning/aggregators
Modified under the GNU General Public License v3.0
'''
import torch
import numpy as np
import copy

device =None
def create_aggregator(aggregator_type, experiment_params, node_hash, logger):
    """
    Create an aggregator based on the aggregator type.

    Args:
        aggregator_type (str): The type of aggregator to create.
        logger (Logger): A logger object to log messages.

    Returns:
        Aggregator: An aggregator object.
    """
    global device
    device = 'cuda:' + str((node_hash %8) ) if torch.cuda.is_available() else 'cpu'
    if aggregator_type == 'fedavg':
        return FedAvg(logger)
    elif aggregator_type == 'median':
        return Median(logger)
    elif aggregator_type == 'krum':
        return Krum(logger)
    elif aggregator_type == 'trimmedmean':
        return TrimmedMean(logger, beta=int(experiment_params['beta']))
    else:
        raise ValueError(f"Aggregator type {aggregator_type} not recognized")
class Aggregator:
    def __init__(self, logger, **kwargs):
        self.logger = logger
        self.kwargs = kwargs
class FedAvg(Aggregator):
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)

    def aggregate(self, models_paths: list, log=True):
        """
        Weighted average of the models.

        Args:
            models: Dictionary with the models (node: model, num_samples).
        """
        if len(models_paths) == 0:
            self.logger.log("[FedAvg] Trying to aggregate models when there are no models")
            return None


        # Total Samples
        total_samples = sum(w for _, w in models_paths)

        # Create a Zero Model
        accum = None

        # Add weighted models
        if log:
            self.logger.log(f"[FedAvg.aggregate] Aggregating models: num={len(models_paths)}")
        for model_path, num_samples in models_paths:
            model = torch.load(model_path, map_location='cpu')
            if accum is None:
                accum = {layer: torch.zeros_like(param).to(device) for layer, param in model.items()}
            for layer in accum:
                accum[layer] += model[layer].to(device)
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
    Median [Dong Yin et al., 2021]
    Paper: https://arxiv.org/pdf/1803.01498.pdf
    """

    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)

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
            self.logger.log(
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

    def aggregate(self, models):
        """
        For each jth model parameter, the master device sorts the jth parameters of
        the m local models and takes the median as the jth parameter
        of the global model. Note that when m is an even number,
        median is the mean of the middle two parameters.

        Args:
            models: Dictionary with the models (node: model,num_samples).
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            self.logger.log(
                               "[Median] Trying to aggregate models when there is no models"
            )
            return None

        models_params = [m for m, _ in models]

        total_models = len(models)

        # Create a Zero Model
        accum = (models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer]).to(device)

        # Add weighteds models
        self.logger.log("[Median.aggregate] Aggregating models: num={}".format(len(models)))

        # Calculate the trimmedmean for each parameter
        for layer in accum:
            weight_layer = accum[layer].to(device)
            # get the shape of layer tensor
            l_shape = list(weight_layer.shape)

            # get the number of elements of layer tensor
            number_layer_weights = torch.numel(weight_layer)
            # if its 0-d tensor
            if l_shape == []:
                weights = torch.tensor([models_params[j][layer].to(device) for j in range(0, total_models)]).to(device)
                weights = weights.double()
                w = self.get_median(weights)
                accum[layer] = w

            else:
                # flatten the tensor
                weight_layer_flatten = weight_layer.view(number_layer_weights)

                # flatten the tensor of each model put all on the same device
                models_layer_weight_flatten = torch.stack([models_params[j][layer].to(device).view(number_layer_weights) for j in range(0, total_models)], 0)
                
                # get the weight list [w1j,w2j,··· ,wmj], where wij is the jth parameter of the ith local model
                median = self.get_median(models_layer_weight_flatten)
                accum[layer] = median.view(l_shape)
        return accum
class Krum(Aggregator):
    """
    Krum [Peva Blanchard et al., 2017]
    Paper: https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html
    """

    def __init__(self, logger, **kwargs):
        super().__init__(logger, **kwargs)

    def aggregate(self, models):
        """
        Krum selects one of the m local models that is similar to other models
        as the global model, the euclidean distance between two local models is used.

        Args:
            models: Dictionary with the models (node: model,num_samples).
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            self.logger.log(
                "[Krum] Trying to aggregate models when there is no models"
            )
            return None


        # Total Samples
        total_samples = sum([y for _, y in models])

        # Create a Zero Model
        accum = copy.deepcopy(models[-1][0])
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer]).to(device)

        # Add weighteds models
        self.logger.log("[Krum.aggregate] Aggregating models: num={}".format(len(models)))

        # Create model distance list
        total_models = len(models)
        distance_list = [0 for i in range(0, total_models)]

        # Calculate the L2 Norm between xi and xj
        min_index = 0
        min_distance_sum = float('inf')

        for i in range(0, total_models):
            m1, _ = models[i]
            for j in range(0, total_models):
                m2, _ = models[j]
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
        m, _ = models[min_index]
        for layer in m:
            accum[layer]=accum[layer].to(device)
            accum[layer] = accum[layer] + m[layer].to(device)
            
        return accum


class TrimmedMean(Aggregator):
    """
    TrimmedMean [Dong Yin et al., 2021]
    Paper: https://arxiv.org/pdf/1803.01498.pdf
    """

    def __init__(self,logger, **kwargs):
        super().__init__(logger)
        self.beta = kwargs.get("beta", 0)

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
            self.logger.log(
                "[TrimmedMean] Trying to aggregate models when there is no models"
            )
            return None

        if weight_len <= 2 * self.beta:
            # logging.error(
            #     "[TrimmedMean] Number of model should larger than 2 * beta"
            # )
            remaining_wrights = weights
            res = torch.mean(remaining_wrights, 0)

        else:
            # remove the largest and smallest β items
            arr_weights = torch.asarray(weights).cpu().numpy()
            # sort the tensor
            arr_weights = np.sort(arr_weights, axis=0)
            nobs = arr_weights.shape[0] # number of observations
            start = self.beta
            end = nobs - self.beta
            atmp = np.partition(arr_weights, (start, end - 1), 0)
            # smallest = torch.topk(arr_weights, k=(self.beta), largest=False, dim=0)
            # largest = torch.topk(arr_weights, k=self.beta, largest=True, dim=0)

            # mask = torch.ones_like(arr_weights, dtype=torch.bool)
            # self.logger.log("mask before" + str(mask) )
            # # mask the smallest and largest elements
            # for i in range(self.beta):  
            #     mask = mask & (arr_weights != smallest.values[i])
            #     mask = mask & (arr_weights != largest.values[i])
            # self.logger.log("mask after" + str(mask) )

            # trimmed = arr_weights[mask]
            # self.logger.log(str(trimmed))


            sl = [slice(None)] * atmp.ndim
            sl[0] = slice(start, end)
            # print num of remaining weights after trimming
            #print(len(atmp[tuple(sl)]))
            #print(atmp[tuple(sl)])
            arr_median = np.mean(atmp[tuple(sl)], axis=0)
            res = torch.tensor(arr_median).to(device)
            # take mean of each column
            #res = torch.mean(trimmed, 0)
            #self.logger.log('res' + str(res))
            #print(res)
        # get the mean of the remaining weights
        return res

    def aggregate(self, models):
        """
        For each jth model parameter, the master device sorts the jth parameters
        of the m local models, i.e., w1j,w2j,··· ,wmj, where wij is the
        jth parameter of the ith local model, removes the largest and
        smallest β of them, and computes the mean of the remaining
        m-2β parameters as the jth parameter of the global model.

        Args:
            models: List of models (node: model, num_samples).
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            self.logger.log(
                "[TrimmedMean] Trying to aggregate models when there is no models"
            )
            return None

        models_params = [m for m, _ in models]

        # Total Samples
        total_samples = sum([y for _, y in models])
        total_models = len(models)

        # Create a Zero Model
        accum = (models[-1][0]).copy()
        for layer in accum:
            accum[layer] = torch.zeros_like(accum[layer])

        # Add weighted models
        self.logger.log("[TrimmedMean.aggregate] Aggregating models: num={}".format(len(models)))

        # Calculate the trimmedmean for each parameter
        for layer in accum:
            weight_layer = accum[layer]
            # get the shape of layer tensor
            l_shape = list(weight_layer.shape)

            # get the number of elements of layer tensor
            number_layer_weights = torch.numel(weight_layer)
            # if its 0-d tensor
            if l_shape == []:
                weights = torch.tensor([models_params[j][layer] for j in range(0, total_models)])
                weights = weights.double()
                w = self.get_trimmedmean(weights)
                accum[layer] = w

            else:
                # flatten the tensor
                weight_layer_flatten = weight_layer.view(number_layer_weights)

                # flatten the tensor of each model
                models_layer_weight_flatten = torch.stack([models_params[j][layer].view(number_layer_weights) for j in range(0, total_models)], 0)

                # get the weight list [w1j,w2j,··· ,wmj], where wij is the jth parameter of the ith local model
                trimmedmean = self.get_trimmedmean(models_layer_weight_flatten)
                #print(l_shape)
                #print(trimmedmean.shape)
                accum[layer] = trimmedmean.view(l_shape)

        return accum