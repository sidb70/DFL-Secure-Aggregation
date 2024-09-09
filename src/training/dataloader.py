from torchvision import datasets, transforms

def load_data(dataset_name: str):
    print("Received dataset name: ", dataset_name)
    if dataset_name == 'mnist':
        return load_mnist()
    elif dataset_name == 'cifar10':
        return load_cifar10()
    else:
        raise ValueError('Dataset not supported')
def load_mnist():
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_dataset = datasets.MNIST(root='./src/training/data', train=True, transform=mnist_transform, download=True)
    return mnist_dataset

def load_cifar10():
    cifar10_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    cifar10_dataset = datasets.CIFAR10(root='./src/training/data', train=True, transform=cifar10_transform, download=True)
    return cifar10_dataset