from .mnist import MNIST_Dataset
from .fmnist import FashionMNIST_Dataset
from .cifar10 import CIFAR10_Dataset



def load_dataset(dataset_name, data_path, normal_class):
    """Loads the dataset."""

    implemented_datasets = ('mnist','fmnist', 'cifar10','my_mnist')
    #assert dataset_name in implemented_datasets

    dataset = None

    if dataset_name == 'mnist':
        dataset = MNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'fmnist':
        dataset = FashionMNIST_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'cifar10':
        dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)

    if dataset_name == 'my_mnist':
    	dataset = CIFAR10_Dataset(root=data_path, normal_class=normal_class)
    return dataset
