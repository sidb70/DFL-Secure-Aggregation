from .digit_classifier import DigitClassifier
from .cifar10_classifier import CIFAR10Classifier
def load_model(model_name: str):
    if model_name == 'mnist':
        return DigitClassifier
    elif model_name == 'cifar10':
        return CIFAR10Classifier
    else:
        raise ValueError('Model not supported')