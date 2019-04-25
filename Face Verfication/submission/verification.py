from DenseNet import MyDenseNet161
from score import *
from ResNet import easyresnet
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
import torch.nn.functional as F
import torchvision
from MyLoss import *
import torch.utils.data as Data
# from torch.optim import lr_scheduler
import copy
import time
import os
import pandas as pd


class ImageLoader(Data.Dataset):
    def __init__(self, list_file, is_test=False):
        self.list_file = open(list_file, 'r').readlines()
        self.is_test = is_test

    def __getitem__(self, index): 
        # should be a string look like '0,0,2\n'
        one_line_content = self.list_file[index]
        if self.is_test:
            image = self._rearange_test_data(one_line_content)
        else:
            image_1, image_2 = self._rearange_data(one_line_content)
        return image

    def _rearange_data(self, one_line_content):
        location = "./validation_verification/"
        image_1, image_2, score = one_line_content.strip('\n').split()
        image_1 = plt.imread(os.path.join(location, image_1)).transpose(2,0,1)/255
        image_2 = plt.imread(os.path.join(location, image_2)).transpose(2,0,1)/255
        return image_1.astype(np.float32), image_2.astype(np.float32), int(score)

    def _rearange_test_data(self, one_line_content):
        location = "./test_veri_T_new/"
        image_1, image_2 = one_line_content.strip('\n').split()
        image_1 = plt.imread(os.path.join(location, image_1)).transpose(2,0,1)/255
        image_2 = plt.imread(os.path.join(location, image_2)).transpose(2,0,1)/255
        return image_1.astype(np.float32), image_2.astype(np.float32)

    def __len__(self):
        return len(self.list_file)


# batch_size = 128
# verification_loader = Data.DataLoader(ImageLoader("trials_test_new.txt", is_test=True),
#                                     batch_size=batch_size, shuffle=False)
# for i in verification_loader:
#     print(i[0])
#     print(i[1].size())


def write_csv(prediction):
    dataframe = pd.read_table("trials_test_new.txt", delim_whitespace=False, header=None, names=['trial'])
    dataframe['score'] = prediction
    dataframe.set_index('trial',inplace=True)
    dataframe.to_csv('prediction.csv', float_format='%.3f')
    print(dataframe.head())


def get_verification(model, test_loader):
    model.eval()
    num_samples = 0
    prediction = np.empty((0,), dtype=float)
    num_iters = len(test_loader)

    for i, (input_1, input_2) in enumerate(test_loader):
        # inputs = inputs.float()
        input_1 = input_1.to(device)
        input_2 = input_2.to(device)
        print("Batch [{}/{}]".format(i+1, num_iters))

        with torch.no_grad():
            output_1 = model(input_1)
            output_1 = F.relu(output_1, inplace=True)
            output_1 = F.adaptive_avg_pool2d(output_1, (1, 1)).view(output_1.size(0), -1)

            output_2 = model(input_2)
            output_2 = F.relu(output_2, inplace=True)
            output_2 = F.adaptive_avg_pool2d(output_2, (1, 1)).view(output_2.size(0), -1)

            outputs = F.cosine_similarity(output_1, output_2, dim=1)
            # print(outputs[0])
            # print(outputs)

            # print(preds.size())
            num_samples += input_1.size(0)
            prediction = np.append(prediction, outputs)
            # print(prediction)

    return np.array(prediction)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Using device: ", torch.cuda.get_device_name(0))

    # train_txt = "train_order_classification_large.txt"
    # validation_txt = "validation_order_classification_large.txt"
    batch_size = 512
    num_classes = 4300
    classifier = MyDenseNet161(num_classes=num_classes)
    # classifier.fc2 = AngleLinear(2208, 2300)
    classifier.load_state_dict(torch.load('./Models/DenseNet_epoch_center1.ckpt'))
    verification = classifier.features.to(device)
    verification_test_loader = Data.DataLoader(ImageLoader("trials_test_new.txt", is_test=True),
                                    batch_size=batch_size, shuffle=False)
    # prediction = np.array([0.354,0.851,0.945])
    prediction = get_verification(verification, verification_test_loader)
    write_csv(prediction)


