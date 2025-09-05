from utils.data_utils import set_seed, get_device, AverageMeter 
import os, torch, shutil
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import matplotlib.pyplot as plt



class Trainer:
    """
    base trainer class
    Specific training loops handled by subclasses 
    """

    def __init__(self, config, output_dir=None, device=None):
        self.config = config
        self.num_workers = self.config.train.num_workers
        self.batch_size = self.config.train.batch_size
        self.lr = self.config.train.lr
        self.n_epochs = self.config.train.n_epochs
        self.dataset = self.config.data.dataset

        set_seed(seed=42)  # do not change seed for reproducibility

        # get device based on availability. 
        if device is None:
            self.device = get_device()
        else:
            self.device = device

        self.data_dir = "./data"
        os.makedirs(self.data_dir, exist_ok=True)

        if output_dir is None:
            self.output_dir = f"./outputs/{self.config.network.model.lower()}"
        else:
            self.output_dir = output_dir

        # if os.path.exists(self.output_dir) and os.path.isdir(self.output_dir):
        #     shutil.rmtree(self.output_dir)
       
        os.makedirs(self.output_dir, exist_ok=True,)

        # dummy transformation so we can download data.
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])

        # Initialize datasets
        if self.dataset.lower() == 'mnist':
            self.trainset = datasets.MNIST(root=self.data_dir, 
                                           train=True, 
                                           download=False, 
                                           transform=transform)
            self.testset = datasets.MNIST(root=self.data_dir, 
                                          train=False, 
                                          download=False, 
                                          transform=transform)
        elif self.dataset.lower() in ['fashionmnist']:
            self.trainset = datasets.FashionMNIST(root=self.data_dir, 
                                                  train=True, 
                                                  download=False, 
                                                  transform=transform)
            self.testset = datasets.FashionMNIST(root=self.data_dir, 
                                                 train=False, 
                                                 download=False, 
                                                 transform=transform)
        else:
           print('unsupported dataset, either use mnist or fashionmnist')
           
        self.train_loader = DataLoader(self.trainset, 
                                       batch_size=self.batch_size, 
                                       shuffle=True, 
                                       num_workers=self.num_workers)
        self.test_loader = DataLoader(self.testset, 
                                      batch_size=self.batch_size, 
                                      shuffle=False, 
                                      num_workers=self.num_workers)

        # get sample data to dynamically get shape incase dataset changes.
        dummy_iterator = iter(self.train_loader)
        sample_input, _ = next(dummy_iterator)
        assert sample_input.dim() == 4, "data shape not expected. You are doing something wrong wrt setup or dataset"
        _, _, self.height, self.width = sample_input.size()
        self.input_dim = int(self.height * self.width)
        self.input_shape_dim = sample_input.dim()

        # sample fixed eval batch and leave on device. not good practice if want to saturate batch_size but ok for us. 
        self.fixed_eval_batch, _ = self.get_fixed_samples(self.testset,n_samples=8)
        self.fixed_eval_batch.to(self.device)

    def _init_optimizer(self, net):
        if self.config.optimizer.type.lower() == 'sgd':
            optimizer = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=0.9, weight_decay=self.config.optimizer.weight_decay)
        elif self.config.optimizer.type.lower() == 'adamw':
            optimizer = torch.optim.AdamW(net.parameters(), lr = self.lr, betas=(0.9,0.999), weight_decay=self.config.optimizer.weight_decay)
        else:
            raise ValueError("unsupported optimizer. use sgd or adamw")
        return optimizer

    @staticmethod
    def get_fixed_samples(dataset, n_samples=8, start_idx=100):
        """
        sample n_samples from the dataset from start_idx.
        """
        images, labels = [], []
        for i in range(start_idx, start_idx + n_samples):
            image, label = dataset[i]
            images.append(image.clone())
            labels.append(label)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels

    @staticmethod
    def save_model(model, model_path):
        model_s = torch.jit.script(model)
        model_s.save(model_path)
        print(f"Model saved to {model_path}")

    @staticmethod
    def load_model(model_path, map_location='cpu'):
        model = torch.jit.load(model_path, map_location=map_location)
        #model.load_state_dict(torch.load(model_path, map_location='cpu'))
        return model

    def train(self):
        raise NotImplementedError

    def evaluate(self, epoch):
        raise NotImplementedError
    

