import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


# batch_size = 1

# transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307),(0.3081))])
# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# #Data set
# train_dataset = torchvision.datasets.MNIST(root='./data',
#                                            train=True,
#                                            transform=transforms.ToTensor(),
#                                            download=True)

# test_dataset = torchvision.datasets.MNIST(root='./data',
#                                           train=False,
#                                           transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


# class GetSamplesFromTwoSets():
#     def __init__(self, train_dataset, number_label):
#         self.train_dataset = train_dataset
#         self.number_label = number_label
#         # assert Inital_set_or_Terminal_set in ["Initial_set", "Terminal_set"], f"Got {Inital_set_or_Terminal_set}, Please indicate Initial set Or Terminal set"

#     def calculate_number_of_samples(self):
#         num_of_samples_for_label = 0
#         for i in self.train_dataset.targets:
#             if i == self.number_label:
#                 num_of_samples_for_label += 1
#         return num_of_samples_for_label

#     def construct_set_of_the_same_label(self):
#         index = 0
#         set_of_samples = torch.zeros(self.calculate_number_of_samples(),
#                                      self.train_dataset.data.shape[1] * self.train_dataset.data.shape[2])
#         for i in range(self.train_dataset.data.shape[0]):
#             if self.train_dataset.targets[i] == self.number_label:
#                 set_of_samples[index, :] = self.train_dataset.data[i, :, :].view(1, -1)
#                 index += 1
#         return set_of_samples, set_of_samples[0:np.int(set_of_samples.shape[0] / 2), :], set_of_samples[np.int(
#             set_of_samples.shape[0] / 2):set_of_samples.shape[0], :]

#     #_, Initial_set, Terminal_set = self.construct_set_of_the_same_label()
#     def get_samples_from_Initial_set_or_Terminal_set(self, Initial_set_or_Terminal_set, num_of_samples):
#         if Initial_set_or_Terminal_set == "Initial_set":
#             _, sample_set, _ = self.construct_set_of_the_same_label()
#         elif Initial_set_or_Terminal_set == "Terminal_set":
#             _, _, sample_set = self.construct_set_of_the_same_label()
#         else:
#             raise ValueError(f"wrong set name, get {Initial_set_or_Terminal_set}, but expecting [Initial_set, Terminal_set]")

#         out = torch.empty(num_of_samples, sample_set.shape[1])
#         index = 0
#         for i in range(num_of_samples):
#             rand_index = torch.randint(0, sample_set.shape[0], (1, 1))
#             out[index, :] = sample_set[rand_index]
#             index += 1
#         return out, sample_set


def samples_from_Terminal_set(num_plot_samples, Terminal_set):
    out = torch.empty(num_plot_samples, Terminal_set.shape[1])
    index = 0
    for i in range(num_plot_samples):
        rand_index = torch.randint(0, Terminal_set.shape[0], (1, 1))
        out[index, :] = Terminal_set[rand_index]
        index += 1
    return out
