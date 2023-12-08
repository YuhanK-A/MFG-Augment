import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F


# print('==> Preparing data..')
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
# ])

# trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
# testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

class DataPreprocessing():
    def __init__(self, trainset):
        self.trainset = trainset

    def calculate_number_of_samples(self, label):
        num_of_samples_for_label = 0
        for i in self.trainset.targets:
            if i == label:
                num_of_samples_for_label += 1
        return num_of_samples_for_label

    def set_of_samples_with_the_same_label(self, label):
        set = torch.zeros(self.calculate_number_of_samples(label), 1, self.trainset.data.shape[1], self.trainset.data.shape[2])
        index_in_set = 0
        for train_index in range(self.trainset.targets.shape[0]):
            if self.trainset.targets[train_index] == label:
                set[index_in_set, 0, :, :] = self.trainset[train_index][0]
                index_in_set += 1
        return set

    def set_of_samples_for_all(self):
        set = torch.zeros(self.trainset.data.shape[0], 1 , self.trainset.data.shape[1], self.trainset.data.shape[2])
        index_in_set = 0
        for train_index in range(self.trainset.data.shape[0]):
            set[index_in_set, 0, :, :] = self.trainset[train_index][0]
            index_in_set += 1
        return set


    def two_sets(self, label):
        set = self.set_of_samples_with_the_same_label(label)
        return set[0:np.int(set.shape[0] / 2), :, :, :], set[np.int(set.shape[0] / 2):set.shape[0], :, :, :]

def divide_feature_vector_set_into_two_sets(set):
    return set[0:np.int(set.shape[0]/2), :], set[np.int(set.shape[0]/2):set.shape[0], :]

def generate_trainset_with_certain_numeber_of_data(trainset, my_train_size):
    my_indexes = list(range(len(trainset)))
    my_train_indexes = my_indexes[:my_train_size]
    my_trainset = torch.utils.data.sampler.SubsetRandomSampler(my_train_indexes)
    return my_trainset

#my_trainset = generate_trainset_with_certain_numeber_of_data(trainset, 10000)

def calculate_number_of_samples_with_same_labels_in_subset(trainset, size, label):
    num_of_samples_for_label = 0
    for i in range(size):
        if trainset.targets[i] == label:
            num_of_samples_for_label += 1
    return  num_of_samples_for_label


def get_features_for_a_certain_label(trainset, feature_set, label):
    num_of_samples_for_label = 0
    for i in range(feature_set.shape[0]):
        if trainset.targets[i] == label:
            num_of_samples_for_label += 1
    index = 0
    out = torch.zeros(num_of_samples_for_label, feature_set.shape[1])
    for j in range(feature_set.shape[0]):
        if trainset.targets[j] == label:
            out[index, :] = feature_set[j, :]
            index += 1
    return out

def get_augmented_images(set_of_feature_vectors_for_all, original_datasize, label, trainset, device, generator, decoder):
    original_feature_set = set_of_feature_vectors_for_all[0:original_datasize, :]
    set_for_a_certain_label_in_subset = get_features_for_a_certain_label(trainset, original_feature_set, label)
    number_of_features_with_same_label = calculate_number_of_samples_with_same_labels_in_subset(trainset, original_datasize, label)
    num_time_slots = 1
    num_augmented_images_for_certain_label = int(num_time_slots * number_of_features_with_same_label * (number_of_features_with_same_label - 1) / 2)
    set_of_augmented_images_for_certain_label = torch.zeros(num_augmented_images_for_certain_label,trainset.data.shape[1], trainset.data.shape[2])
    label_set = torch.zeros(num_augmented_images_for_certain_label, 1)
    index = 0
    for i in range(number_of_features_with_same_label - 1):
        number_of_samples = number_of_features_with_same_label - i - 1
        Terminal_set_for_data_augmentation = set_for_a_certain_label_in_subset[i + 1:number_of_features_with_same_label,:].to(device)
        sample = set_for_a_certain_label_in_subset[i, :].unsqueeze(0)
        Initial_set_for_data_augmentation = sample.repeat(number_of_samples, 1).to(device)
        # augmented_data_1 = generator(torch.tensor([0.25], dtype=torch.float).to(device).expand(number_of_samples, 1),
        #                              Initial_set_for_data_augmentation,
        #                              Terminal_set_for_data_augmentation).cpu().detach()
        augmented_data_2 = generator(torch.tensor([0.5], dtype=torch.float).to(device).expand(number_of_samples, 1),
                                     Initial_set_for_data_augmentation,
                                     Terminal_set_for_data_augmentation).cpu().detach()
        # augmented_data_3 = generator(torch.tensor([0.75], dtype=torch.float).to(device).expand(number_of_samples, 1),
        #                              Initial_set_for_data_augmentation,
        #                              Terminal_set_for_data_augmentation).cpu().detach()
        # augmented_datas = torch.cat((augmented_data_1, augmented_data_2, augmented_data_3), 0)
        augmented_datas = augmented_data_2
        for j in range(augmented_datas.shape[0]):
            rec_img = decoder(torch.reshape(augmented_datas[j], (1, 320)))
            set_of_augmented_images_for_certain_label[index, :,:] = rec_img.detach().cpu().squeeze()
            label_set[index, 0] = label
            index += 1
            #plt.imshow(rec_img.detach().cpu().squeeze().numpy(), cmap='gist_gray')
            #plt.show()
    return set_of_augmented_images_for_certain_label, label_set

def get_augmented_images_CIFAR(set_of_feature_vectors_for_all, original_datasize, label, trainset, device, generator, decoder):
    original_feature_set = set_of_feature_vectors_for_all[0:original_datasize, :]
    set_for_a_certain_label_in_subset = get_features_for_a_certain_label(trainset, original_feature_set, label)
    number_of_features_with_same_label = calculate_number_of_samples_with_same_labels_in_subset(trainset, original_datasize, label)
    ## 简略版
    Ini_set = set_for_a_certain_label_in_subset[:int(number_of_features_with_same_label/2),:].to(device)
    Termi_set = set_for_a_certain_label_in_subset[int(number_of_features_with_same_label/2):,:].to(device)
    out11 = torch.zeros(1,3,32,32).to(device)
    out11_labels = torch.zeros(1, 1, dtype=int).to(device)

    if number_of_features_with_same_label%2 !=0:
        Ini_set = set_for_a_certain_label_in_subset[:int(number_of_features_with_same_label/2)+1,:].to(device)
    num = Termi_set.shape[0]
    aug_data1 = generator(torch.tensor([0.15], dtype=torch.float).to(device).expand(num, 1), Ini_set, Termi_set)
    aug_data2 = generator(torch.tensor([0.85], dtype=torch.float).to(device).expand(num, 1), Ini_set, Termi_set)
    augmented_datas = torch.cat((aug_data1, aug_data2), 0)
    index = 0
    for j in range(augmented_datas.shape[0]):
        rec_img = decoder(torch.reshape(augmented_datas[j], (1, 48, 4, 4)).to(device))
        out11 = torch.cat((out11,rec_img), 0)
        my_label = torch.tensor([[label]], dtype=int).to(device)
        out11_labels = torch.cat((out11_labels, my_label), 0)
        index += 1

    





    ## 完全版
    # num_time_slots = 2
    # num_augmented_images_for_certain_label = int(num_time_slots * number_of_features_with_same_label * (number_of_features_with_same_label - 1) / 2)
    # set_of_augmented_images_for_certain_label = torch.zeros(num_augmented_images_for_certain_label, 3, 32, 32)
    # label_set = torch.zeros(num_augmented_images_for_certain_label, 1)
    # index = 0
    # for i in range(number_of_features_with_same_label - 1):
    #     number_of_samples = number_of_features_with_same_label - i - 1
    #     Terminal_set_for_data_augmentation = set_for_a_certain_label_in_subset[i + 1:number_of_features_with_same_label,:].to(device)
    #     sample = set_for_a_certain_label_in_subset[i, :].unsqueeze(0)
    #     Initial_set_for_data_augmentation = sample.repeat(number_of_samples, 1).to(device)
    #     augmented_data_1 = generator(torch.tensor([0.15], dtype=torch.float).to(device).expand(number_of_samples, 1),
    #                                   Initial_set_for_data_augmentation,
    #                                   Terminal_set_for_data_augmentation).cpu().detach()
    #     augmented_data_2 = generator(torch.tensor([0.85], dtype=torch.float).to(device).expand(number_of_samples, 1),
    #                                  Initial_set_for_data_augmentation,
    #                                  Terminal_set_for_data_augmentation).cpu().detach()
    #     # augmented_data_3 = generator(torch.tensor([0.75], dtype=torch.float).to(device).expand(number_of_samples, 1),
    #     #                              Initial_set_for_data_augmentation,
    #     #                              Terminal_set_for_data_augmentation).cpu().detach()
    #     augmented_datas = torch.cat((augmented_data_1, augmented_data_2), 0)
    #     for j in range(augmented_datas.shape[0]):
    #         rec_img = decoder(torch.reshape(augmented_datas[j], (1, 48, 4, 4)).to(device))
    #         set_of_augmented_images_for_certain_label[index, :, :, :] = rec_img[0, :, :, :].detach().cpu().squeeze()
    #         label_set[index, 0] = label
    #         index += 1
    #         #plt.imshow(rec_img.detach().cpu().squeeze().numpy(), cmap='gist_gray')
    #         #plt.show()
    # return set_of_augmented_images_for_certain_label, label_set
    return out11[1:,:,:,:], out11_labels[1:,:]



def generate_augmented_image_set_for_certain_subset(trainset, original_datasize, set_of_feature_vectors_for_all, device, generator, decoder):
    out = torch.zeros(1, trainset.data.shape[1], trainset.data.shape[2])
    out_label = torch.zeros(1, 1)
    for label in range(10):
        set, set_label = get_augmented_images(set_of_feature_vectors_for_all, original_datasize, label, trainset, device, generator, decoder)
        out = torch.cat((out, set), 0)
        out_label = torch.cat((out_label, set_label), 0)
    return out[1:, :, :], out_label[1:,:].int().long()

def generate_augmented_image_set_for_certain_subset_CIFAR(trainset, original_datasize, set_of_feature_vectors_for_all, device, generator, decoder):
    out = torch.zeros(1, 3, 32, 32).to(device)
    out_label = torch.zeros(1, 1, dtype=int).to(device)
    for label in range(10):
        set, set_label = get_augmented_images_CIFAR(set_of_feature_vectors_for_all, original_datasize, label, trainset, device, generator, decoder)
        out = torch.cat((out, set), 0)
        out_label = torch.cat((out_label, set_label), 0)
    return out[1:, :, :, :], out_label[1:,:]



class MyMnistDataset(torch.utils.data.Dataset):
    def __init__(self, trainset, original_datasize, transform):
        self.original_datasize = original_datasize
        self.trainset = trainset.data[:self.original_datasize,:,:]
        self.targets = trainset.targets[:self.original_datasize]
        self.transform = transform

    def __getitem__(self, index):
        image = self.trainset[index]
        image = Image.fromarray(image.numpy(), mode='L')
        image = self.transform(image)
        label = self.targets[index]
        out = torch.zeros(1, 28, 28)
        out[0,:,:] = image
        return out, label

    def __len__(self):
        return self.trainset.shape[0]


class MyCifar10ImageTrainSet(torch.utils.data.Dataset):
    def __init__(self, trainset, original_datasize, transform):
        self.original_datasize = original_datasize
        self.trainset = trainset.data[:self.original_datasize,:,:, :]
        self.targets = trainset.targets[:self.original_datasize]
        self.transform = transform

    def __getitem__(self, index):
        image = self.trainset[index]
        image = Image.fromarray(image)
        image = self.transform(image)
        label = self.targets[index]

        return image, label

    def __len__(self):
        return self.trainset.shape[0]


class MyMnistFeatureset(torch.utils.data.Dataset):
    def __init__(self, feature_vector_for_all, original_datasize, trainset):
        self.feature_set_for_certain_size = feature_vector_for_all[0:original_datasize, :]
        self.targets = trainset.targets[0:original_datasize]

    def __getitem__(self, index):
        data = self.feature_set_for_certain_size[index]
        label = self.targets[index]
        return data, label

    def __len__(self):
        return self.targets.shape[0]


class MyCifarFeatureTrainset(torch.utils.data.Dataset):
    def __init__(self, feature_vector_for_all, original_datasize, trainset):
        self.feature_set_for_certain_size = feature_vector_for_all[0:original_datasize, :]
        self.targets = trainset.targets[0:original_datasize]

    def __getitem__(self, index):
        data = self.feature_set_for_certain_size[index]
        label = self.targets[index]
        return data, label

    def __len__(self):
        return len(self.targets)


class MyMnistFeatureTestSet(torch.utils.data.Dataset):
    def __init__(self, feature_test_set, test_set):
        self.feature_test_set = feature_test_set
        self.targets = test_set.targets

    def __getitem__(self, index):
        data = self.feature_test_set[index]
        label = self.targets[index]
        return data, label

    def __len__(self):
        return self.targets.shape[0]

class MyCifarFeatureTestset(torch.utils.data.Dataset):
    def __init__(self, feature_test_set, test_set):
        self.feature_test_set = feature_test_set
        self.targets = test_set.targets

    def __getitem__(self, index):
        data = self.feature_test_set[index]
        label = self.targets[index]
        return data, label

    def __len__(self):
        return len(self.targets)

class Yuhan_lin_layer(nn.Module):
    def __init__(self, dim_feature=320, num_classes=10, dropout_rate= 0.2):
        super(Yuhan_lin_layer, self).__init__()
        self.dropout_rate = dropout_rate
        self.linear = nn.Linear(dim_feature, num_classes)

    def forward(self, x):
        out = x
        dropout_rate = self.dropout_rate
        if self.training and dropout_rate > 0:
            out = F.dropout(x, p=dropout_rate)
        out = self.linear(out)
        return out
    
def get_augmented_images_cross_type(set_of_feature_vectors_for_all, original_datasize, trainset, device, generator, decoder):
    original_feature_set = set_of_feature_vectors_for_all[0:original_datasize, :]
    aug_imgs = torch.zeros(1, trainset.data.shape[1], trainset.data.shape[2])
    aug_labels = torch.zeros(1, 1, dtype=int)
    for label1 in range(9):
        for label2 in range(label1+1,10):
            num1 = calculate_number_of_samples_with_same_labels_in_subset(trainset, original_datasize, label1)
            num2 = calculate_number_of_samples_with_same_labels_in_subset(trainset, original_datasize, label2)
            set1 = get_features_for_a_certain_label(trainset, original_feature_set, label1)
            set2 = get_features_for_a_certain_label(trainset, original_feature_set, label2)
            for i in range(num1):
                sample = set1[i,:].unsqueeze(0)
                Ini_set = sample.repeat(num2, 1).to(device)
                Termi_set = set2.to(device)
                aug_feature1 = generator(torch.tensor([0.23], dtype=torch.float).to(device).expand(num2, 1), Ini_set, Termi_set).cpu().detach()
                aug_feature2 = generator(torch.tensor([0.77], dtype=torch.float).to(device).expand(num2, 1), Ini_set, Termi_set).cpu().detach()
                aug_label1 = torch.tensor([[label1]], dtype=int)
                aug_label2 = torch.tensor([[label2]], dtype=int)

                for p in range(aug_feature1.shape[0]):
                    rec_img = decoder(torch.reshape(aug_feature1[p], (1, 320)))
                    aug_imgs = torch.cat((aug_imgs, rec_img.detach().cpu().squeeze(0)), 0)
                    aug_labels = torch.cat((aug_labels, aug_label1), 0)

                for q in range(aug_feature2.shape[0]):
                    rec_img = decoder(torch.reshape(aug_feature2[q], (1, 320)))
                    aug_imgs = torch.cat((aug_imgs, rec_img.detach().cpu().squeeze(0)), 0)
                    aug_labels = torch.cat((aug_labels, aug_label2), 0)

    return aug_imgs[1:, :, :], aug_labels[1:, :]

def get_augmented_feature_cross_type(set_of_feature_vectors_for_all, original_datasize, trainset, device, generator):
    feature_set_cert_size = set_of_feature_vectors_for_all[0:original_datasize, :]
    aug_features = torch.zeros(1, 320)
    aug_fea_labels = torch.zeros(1, 1, dtype=int)
    for label1 in range(9):
        for label2 in range(label1+1,10):
            num1 = calculate_number_of_samples_with_same_labels_in_subset(trainset, original_datasize, label1)
            num2 = calculate_number_of_samples_with_same_labels_in_subset(trainset, original_datasize, label2)
            set1 = get_features_for_a_certain_label(trainset, feature_set_cert_size, label1)
            set2 = get_features_for_a_certain_label(trainset, feature_set_cert_size, label2)
            for i in range(num1):
                sample = set1[i,:].unsqueeze(0)
                Ini_set = sample.repeat(num2, 1).to(device)
                Termi_set = set2.to(device)
                aug_feature1 = generator(torch.tensor([0.23], dtype=torch.float).to(device).expand(num2, 1), Ini_set, Termi_set).cpu().detach()
                aug_feature2 = generator(torch.tensor([0.77], dtype=torch.float).to(device).expand(num2, 1), Ini_set, Termi_set).cpu().detach()
                aug_label1 = torch.tensor([[label1]], dtype=int)
                aug_label2 = torch.tensor([[label2]], dtype=int)

                aug_features = torch.cat((aug_features, aug_feature1), dim=0)
                aug_fea_labels = torch.cat((aug_fea_labels, aug_label1.repeat(num2,1)), 0)
                aug_features = torch.cat((aug_features, aug_feature2), dim=0)
                aug_fea_labels = torch.cat((aug_fea_labels, aug_label2.repeat(num2,1)), 0)

    return aug_features[1:, :], aug_fea_labels[1:, :]

def get_augmented_images_cross_type_CIFAR(set_of_feature_vectors_for_all, original_datasize, trainset, device, generator, decoder):
    original_feature_set = set_of_feature_vectors_for_all[0:original_datasize, :]
    aug_imgs = torch.zeros(1, 3, 32, 32)
    aug_labels = torch.zeros(1, 1, dtype=int)
    for label1 in range(9):
        for label2 in range(label1+1,10):
            num1 = calculate_number_of_samples_with_same_labels_in_subset(trainset, original_datasize, label1)
            num2 = calculate_number_of_samples_with_same_labels_in_subset(trainset, original_datasize, label2)
            set1 = get_features_for_a_certain_label(trainset, original_feature_set, label1)
            set2 = get_features_for_a_certain_label(trainset, original_feature_set, label2)
            for i in range(num1):
                sample = set1[i,:].unsqueeze(0)
                Ini_set = sample.repeat(num2, 1).to(device)
                Termi_set = set2.to(device)
                aug_feature1 = generator(torch.tensor([0.23], dtype=torch.float).to(device).expand(num2, 1), Ini_set, Termi_set).cpu().detach()
                aug_feature2 = generator(torch.tensor([0.77], dtype=torch.float).to(device).expand(num2, 1), Ini_set, Termi_set).cpu().detach()
                aug_label1 = torch.tensor([[label1]], dtype=int)
                aug_label2 = torch.tensor([[label2]], dtype=int)

                for p in range(aug_feature1.shape[0]):
                    rec_img = decoder(torch.reshape(aug_feature1[p], (1, 320)))
                    aug_imgs = torch.cat((aug_imgs, rec_img.detach().cpu().squeeze(0)), 0)
                    aug_labels = torch.cat((aug_labels, aug_label1), 0)

                for q in range(aug_feature2.shape[0]):
                    rec_img = decoder(torch.reshape(aug_feature2[q], (1, 320)))
                    aug_imgs = torch.cat((aug_imgs, rec_img.detach().cpu().squeeze(0)), 0)
                    aug_labels = torch.cat((aug_labels, aug_label2), 0)

    return aug_imgs[1:, :, :, :], aug_labels[1:, :]


def get_augmented_features_same_label(set_of_feature_vectors_for_all, original_datasize, label, trainset, device, generator):
    original_feature_set = set_of_feature_vectors_for_all[0:original_datasize, :]
    set_for_a_certain_label_in_subset = get_features_for_a_certain_label(trainset, original_feature_set, label)
    number_of_features_with_same_label = calculate_number_of_samples_with_same_labels_in_subset(trainset, original_datasize, label)
    aug_features = torch.zeros(1, 320)
    aug_labels = torch.zeros(1, 1, dtype=int)
    num_time_slots = 3
    num_labels = int(num_time_slots * number_of_features_with_same_label * (number_of_features_with_same_label - 1) / 2)
    for i in range(number_of_features_with_same_label - 1):
        number_of_samples = number_of_features_with_same_label - i - 1
        Terminal_set_for_data_augmentation = set_for_a_certain_label_in_subset[i + 1:number_of_features_with_same_label,:].to(device)
        sample = set_for_a_certain_label_in_subset[i, :].unsqueeze(0)
        Initial_set_for_data_augmentation = sample.repeat(number_of_samples, 1).to(device)
        augmented_data_1 = generator(torch.tensor([0.25], dtype=torch.float).to(device).expand(number_of_samples, 1),
                                      Initial_set_for_data_augmentation,
                                      Terminal_set_for_data_augmentation).cpu().detach()
        augmented_data_2 = generator(torch.tensor([0.5], dtype=torch.float).to(device).expand(number_of_samples, 1),
                                        Initial_set_for_data_augmentation,
                                        Terminal_set_for_data_augmentation).cpu().detach()
        augmented_data_3 = generator(torch.tensor([0.75], dtype=torch.float).to(device).expand(number_of_samples, 1),
                                     Initial_set_for_data_augmentation,
                                     Terminal_set_for_data_augmentation).cpu().detach()
        aug_label = torch.tensor([[label]], dtype=int)
        aug_features = torch.cat((aug_features, augmented_data_1, augmented_data_2, augmented_data_3), 0)
    
    aug_labels = torch.cat((aug_labels, aug_label.repeat(num_labels, 1)), 0)

    return aug_features[1:,:], aug_labels[1:,:]

def generate_augmented_features_for_50(trainset, original_datasize, set_of_feature_vectors_for_all, device, generator):
    out = torch.zeros(1, 320)
    out_label = torch.zeros(1, 1, dtype=int)
    for label in range(10):
        set, set_label = get_augmented_features_same_label(set_of_feature_vectors_for_all, original_datasize, label, trainset, device, generator)
        out = torch.cat((out, set), 0)
        out_label = torch.cat((out_label, set_label), 0)
    return out[1:, :], out_label[1:,:]
# my_train_set = DataPreprocessing(trainset)
# samples_for_number_5 = my_train_set.set_of_samples_with_the_same_label(5)
# num = my_train_set.calculate_number_of_samples(5)
# Initial_set, Terminal_set = my_train_set.two_sets(5)





