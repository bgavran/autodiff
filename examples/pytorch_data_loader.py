import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


class MNIST:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        train_dataset = dsets.MNIST(root="../data",
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
        test_dataset = dsets.MNIST(root="../data",
                                   train=False,
                                   transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True)
