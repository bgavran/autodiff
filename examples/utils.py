import numpy as np
import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms


# Using PyTorch dataloader to load up MNIST :)

class MNIST:
    def __init__(self, batch_size):
        self.batch_size = batch_size

        train_dataset = dsets.MNIST(root="./data",
                                    train=True,
                                    transform=transforms.ToTensor(),
                                    download=True)
        test_dataset = dsets.MNIST(root="./data",
                                   train=False,
                                   transform=transforms.ToTensor())

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True)

        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                       batch_size=batch_size,
                                                       shuffle=True)


class TextLoader:
    def __init__(self, file):
        self.file = file
        split_point = int(0.75 * len(self.file))
        self.file_train, self.file_test = self.file[:split_point], self.file[split_point:]

        self.chars = sorted(list(set(self.file)))
        self.num_chars = len(self.chars)
        self.char_to_ind = {c: ind for ind, c in enumerate(self.chars)}
        self.ind_to_char = {ind: c for c, ind in self.char_to_ind.items()}

        self.file_train_ind = [self.char_to_ind[char] for char in self.file_train]
        self.file_test_ind = [self.char_to_ind[char] for char in self.file_test]

    def next_batch(self, batch_size, seq_len=20, test=False):
        if test:
            source = self.file_test_ind
        else:
            source = self.file_train_ind
        indices = np.random.randint(0, len(source) - seq_len, batch_size)
        return np.array([source[ind:ind + seq_len] for ind in indices])

    def text_to_indices(self, text):
        return np.expand_dims([self.char_to_ind[char] for char in text], axis=0)

    def to_one_hot(self, batch):
        identity = np.eye(self.num_chars)
        return np.array([identity[b] for b in batch])
