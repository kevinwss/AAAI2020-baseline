from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import MNIST
from base.torchvision_dataset import TorchvisionDataset
from .preprocessing import get_target_label_idx, global_contrast_normalization

import torchvision.transforms as transforms
import numpy as np
import torch

class MNIST_Dataset(TorchvisionDataset):

    def __init__(self, root: str, normal_class=0):
        super().__init__(root)

        self.n_classes = 2  # 0: normal, 1: outlier
        #self.normal_classes = tuple([normal_class])
        #self.outlier_classes = list(range(0, 10))
        #self.outlier_classes.remove(normal_class)

        self.normal_classes = tuple([0,1,2,3,4,5,6,7,8,9])

        self.outlier_classes = []
        # Pre-computed min and max values (after applying GCN) from train data per class
        min_max = [(-0.8826567065619495, 9.001545489292527),
                   (-0.6661464580883915, 20.108062262467364),
                   (-0.7820454743183202, 11.665100841080346),
                   (-0.7645772083211267, 12.895051191467457),
                   (-0.7253923114302238, 12.683235701611533),
                   (-0.7698501867861425, 13.103278415430502),
                   (-0.778418217980696, 10.457837397569108),
                   (-0.7129780970522351, 12.057777597673047),
                   (-0.8280402650205075, 10.581538445782988),
                   (-0.7369959242164307, 10.697039838804978)]

        # MNIST preprocessing: GCN (with L1 norm) and min-max feature scaling to [0,1]
        '''
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),
                                        transforms.Normalize([min_max[normal_class][0]],
                                                             [min_max[normal_class][1] - min_max[normal_class][0]])])
        
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Lambda(lambda x: global_contrast_normalization(x, scale='l1')),])
        '''
        transform = transforms.Compose([transforms.ToTensor()])
        target_transform = transforms.Lambda(lambda x: int(x in self.outlier_classes))


        train_set = MyMNIST(root=self.root, train=True, download=True,
                            transform=transform, target_transform=None)
        # Subset train_set to normal class
        train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        #print("train_idx_normal",train_idx_normal)
        self.train_set = Subset(train_set, train_idx_normal)
        self.train_set = train_set
        self.test_set = MyMNIST(root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)


class MyMNIST(MNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(MyMNIST, self).__init__(*args, **kwargs)
        self.anomaly_rate = 0.10

        def get_anomaly(anomaly_data):
            n_anomaly = len(anomaly_data)
            dim = 28
            #print("anomaly_data",anomaly_data.shape)
            a1,a2 = anomaly_data[:n_anomaly//2,:dim//2,:],anomaly_data[:n_anomaly//2,dim//2:,:]
            b1,b2 = anomaly_data[n_anomaly//2:,:dim//2,:],anomaly_data[n_anomaly//2:,dim//2:,:]

            #print("a1",a1.shape)
            #print("b2",b2.shape)
            anomaly_data1 = np.concatenate((a1,b2),axis = 1)
            anomaly_data2 = np.concatenate((b1,a2),axis = 1)
            anomaly_data = np.concatenate((anomaly_data1,anomaly_data2),axis = 0)
            return anomaly_data

        if not self.train:
            #pass
            test_data_normal = self.test_data[:9000,:,:]
            test_data_anomaly = get_anomaly(self.test_data[9000:,:,:])
            '''
            print("self.test_labels",self.test_labels)
            print("0")
            new_test_labels = [0 for _ in range(10000)]
            for i in range(10000):
                if self.test_labels[i] == 3:

                    new_test_labels[i] = 1
                else:
                    new_test_labels[i] = 0
            print("1")
            self.test_labels = torch.from_numpy(np.array(new_test_labels))
            print("self.test_labels",self.test_labels)
            '''
            self.test_data = torch.from_numpy(np.concatenate((test_data_normal,test_data_anomaly),axis = 0))
            self.test_labels = torch.from_numpy(np.array([0]*(len(test_data_normal)) + [1]*len(test_data_anomaly)))

        else:
            n_train = len(self.train_data)
            n_normal = n_train - int(self.anomaly_rate*n_train)
            normal_train = self.train_data[:n_normal,:,:]
            tobe_anomaly_train = self.train_data[n_normal:,:,:]
            anomaly_train = get_anomaly(tobe_anomaly_train)

            self.train_data = torch.from_numpy(np.concatenate((normal_train,anomaly_train),axis = 0))

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """
        #if not self.train:
        #    print("self.test_data0",self.test_data.shape)
        #    print("self.test_labels0",self.test_labels.shape,self.test_labels)
        #--------------
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]

        else:

            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if self.train:
            img = Image.fromarray(img.numpy(), mode='L')
        else:
            img = Image.fromarray(img.numpy(), mode='L')

        print("img",img)
        if self.transform is not None:
            img = self.transform(img)
        #print(index)
        #print("self.test_data0",self.train_data.shape)
        #print("self.test_labels0",self.test_labels.shape,self.test_labels)
        #if self.target_transform is not None:
        #    target = self.target_transform(target)

        #-------------
        #if not self.train:
        #    print("self.test_data",self.test_data.shape)
        #    print("self.test_labels",target,target)
        #print("target",target)

        return img, target, index  # only line changed
