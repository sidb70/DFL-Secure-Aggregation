'''
Source: https://github.com/enriquetomasmb/fedstellar/blob/main/fedstellar/learning/aggregators/fedavg.py
Modified under the GNU General Public License v3.0
'''
import torch
class FedAvg:
    """
    Federated Averaging (FedAvg) [McMahan et al., 2016]
    Paper: https://arxiv.org/abs/1602.05629
    """

    def __init__(self, logger):
        self.logger=logger

    def aggregate(self, models: list):
        """
        Weighted average of the models.

        Args:
            models: Dictionary with the models (node: model, num_samples).
        """
        if len(models) == 0:
            self.logger.log("[FedAvg] Trying to aggregate models when there are no models")
            return None


        # Total Samples
        total_samples = sum(w for _, w in models)

        # Create a Zero Model
        accum = {layer: torch.zeros_like(param) for layer, param in models[-1][0].items()}

        # Add weighted models
        self.logger.log(f"[FedAvg.aggregate] Aggregating models: num={len(models)}")
        for model, weight in models:
            for layer in accum:
                accum[layer] += model[layer] * weight

        # Normalize Accum
        for layer in accum:
            accum[layer] /= total_samples
            
        # self.print_model_size(accum)

        return accum