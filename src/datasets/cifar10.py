from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms
import numpy as np
import torch

class CIFAR10_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=5):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        '''
        self.normal_classes = tuple([normal_class])
        self.outlier_classes = list(range(0, 10))
        self.outlier_classes.remove(normal_class)
        '''
        self.normal_classes = tuple([0,1,2,3,4,5,6,7,8,9])
        self.outlier_classes = []

        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-28.94083453598571, 13.802961825439636),
                   (-6.681770233365245, 9.158067708230273),
                   (-34.924463588638204, 14.419298165027628),
                   (-10.599172931391799, 11.093187820377565),
                   (-11.945022995801637, 10.628045447867583),
                   (-9.691969487694928, 8.948326776180823),
                   (-9.174940012342555, 13.847014686472365),
                   (-6.876682005899029, 12.282371383343161),
                   (-15.603507135507172, 15.2464923804279),
                   (-6.132882973622672, 8.046098172351265)]

        # CIFAR-10 preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        '''
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]] * 3,
                                                             [min_max[normal_class][1] - min_max[normal_class][0]] * 3)])
        '''

        transform = transforms.Compose([transforms.ToTensor()])
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))

        train_set = MyCIFAR10(root=self.root, train=True, download=True,
                              transform=transform, target_transform=target_transform)
        # Subset train set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels, self.normal_classes)
        #self.train_set = Subset(train_set, train_idx_normal)
        self.train_set = train_set

        self.test_set = MyCIFAR10(root=self.root, train=False, download=True,
                                  transform=transform, target_transform=target_transform)


class MyCIFAR10(CIFAR10):
    """Torchvision CIFAR10 class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyCIFAR10, self).__init__(*args, **kwargs)

        self.anomaly_rate = 0.05

        def get_anomaly(anomaly_data):
            n_anomaly = len(anomaly_data)
            dim = 32
            print("anomaly_data",anomaly_data.shape)
            a1,a2 = anomaly_data[:n_anomaly//2,:dim//2,:,:],anomaly_data[:n_anomaly//2,dim//2:,:,:]
            b1,b2 = anomaly_data[n_anomaly//2:,:dim//2,:,:],anomaly_data[n_anomaly//2:,dim//2:,:,:]

            #print("a1",a1.shape)
            #print("b2",b2.shape)
            anomaly_data1 = np.concatenate((a1,b2),axis = 1)
            anomaly_data2 = np.concatenate((b1,a2),axis = 1)
            anomaly_data = np.concatenate((anomaly_data1,anomaly_data2),axis = 0)
            return anomaly_data

        if not self.train:
            #pass
            test_data_normal = self.test_data[:9000,:,:,:]
            test_data_anomaly = get_anomaly(self.test_data[9000:,:,:,:])
            #self.test_data = torch.from_numpy(np.concatenate((test_data_normal,test_data_anomaly),axis = 0))
            #self.test_labels = torch.from_numpy(np.array([0]*(len(test_data_normal)) + [1]*len(test_data_anomaly)))

            self.test_data = np.concatenate((test_data_normal,test_data_anomaly),axis = 0)
            self.test_labels = np.array([0]*(len(test_data_normal)) + [1]*len(test_data_anomaly))
        else:
            n_train = len(self.train_data)
            n_normal = n_train - int(self.anomaly_rate*n_train)
            normal_train = self.train_data[:n_normal,:,:,:]
            tobe_anomaly_train = self.train_data[n_normal:,:,:,:]
            anomaly_train = get_anomaly(tobe_anomaly_train)

            #self.train_data = torch.from_numpy(np.concatenate((normal_train,anomaly_train),axis = 0))
            self.train_data = np.concatenate((normal_train,anomaly_train),axis = 0)

    def __getitem__(self, index):
        """Override the original method of the CIFAR10 class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        #img = Image.fromarray(img.numpy(), mode='L')
        if self.transform is not None:
            img = self.transform(img)

        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        return img, target, index  # only line changed
