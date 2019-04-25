from DenseNet import MyDenseNet161
from ResNet import easyresnet
import numpy as np 
import matplotlib.pyplot as plt 
import torch
import torch.nn as nn 
import torchvision
from MyLoss import *
import torch.utils.data as Data
from torch.optim import lr_scheduler
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
            return image
        else:
            image, label = self._rearange_data(one_line_content)
            return image, int(label)

    def _rearange_data(self, one_line_content):
        image, label = one_line_content.strip('\n').split(';')
        image = plt.imread(image).transpose(2,0,1)/255
        return image, label

    def _rearange_test_data(self, one_line_content):
        image = one_line_content.strip('\n')
        # print(os.path.join('./test_classification/medium', image))
        image = plt.imread(os.path.join('./test_classification/medium', image)).transpose(2,0,1)/255
        return image

    def __len__(self):
        return len(self.list_file)


def train_classifier(Model, klloss, centerloss, optimizer1, optimizer2, scheduler1, scheduler2, num_epochs=10):
    since = time.time()
    best_model_wts = copy.deepcopy(Model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch+1, num_epochs))
        print("-"*30)
        scheduler1.step()
        scheduler2.step()

        for phase in ["train", "validation"]:
            if phase == "train":
                Model.train()
            else:
                Model.eval()

            running_loss = 0.0
            epoch_corrects = 0
            num_samples = 0

            for i, (inputs, labels) in enumerate(dataloader_dict[phase]):
                inputs = inputs.float()
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    features, outputs = Model(inputs)
                    loss = klloss(outputs, labels) + centerloss(features, labels) * 0.05
                    # lossd = loss.data[0]
                    outputs = nn.functional.softmax(outputs, dim=1)
                    _, preds = torch.max(outputs, 1)
                    if phase == "train":
                        loss.backward()
                        optimizer1.step()
                        optimizer2.step()

                print('Batch [{}/{}], [{}]--Batch Loss: {:.4f}'.format(i+1, len(dataloader_dict[phase]), phase, loss.data[0]))

                num_samples  += inputs.size(0)
                running_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)
            epoch_loss = running_loss/num_samples 
            epoch_acc = epoch_corrects.double()/num_samples
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            print("-"*30)
            torch.save(Model.state_dict(), './Models/DenseNet_epoch_center{}.ckpt'.format(epoch+1))
            torch.save(centerloss.state_dict(), './Models/DenseNet_centerloss.ckpt')
            
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(Model.state_dict())
        print()
        
    time_elapsed = time.time()-since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    Model.load_state_dict(best_model_wts)
    return Model


def get_prediction(model, test_loader):
    model.eval()
    num_samples = 0
    prediction = np.empty((0,), dtype=int)
    num_iters = len(test_loader)

    for i, inputs in enumerate(test_loader):
        inputs = inputs.float()
        inputs = inputs.to(device)
        print("Batch [{}/{}]".format(i+1, num_iters))

        with torch.no_grad():
            outputs = model(inputs)
            # print(type(outputs))

            outputs = nn.functional.softmax(outputs, dim=1)
            # print(outputs)

            preds = my_max(outputs)
            print(preds)
            # print(torch.max(outputs,1))
            # print(_)

            # print(preds.size())
            num_samples += inputs.size(0)
            prediction = np.append(prediction, preds)

    return np.array(prediction, dtype=int)


def my_max(tensor):
    top_k_value, top_k_index = torch.topk(tensor, 3, dim=1)
    i = 0
    max_index = np.zeros(tensor.size(0))
    for i in range(tensor.size(0)):
        if top_k_index[i,0] < 2300:
            max_index[i] = top_k_index[i,0]
        elif top_k_index[i,1] < 2300:
            max_index[i] = top_k_index[i,1]
        else:
            max_index[i] = top_k_index[i,2]
        i += 1
    return max_index


def write_csv(prediction):
    dataframe = pd.DataFrame(np.arange(prediction.shape[0]), columns=['id'])
    dataframe['label'] = prediction
    dataframe.set_index('id',inplace=True)
    dataframe.to_csv('prediction.csv')
    print(dataframe.head())


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Using device: ", torch.cuda.get_device_name(0))

    train_txt = "train_order_classification_large.txt"
    validation_txt = "validation_order_classification_large.txt"
    batch_size = 128
    num_classes = 4300
    Loader_num_workers = 1 
    train_loader = Data.DataLoader(ImageLoader(train_txt, is_test=False),
                                    batch_size=batch_size, shuffle=True)
    validation_loader = Data.DataLoader(ImageLoader(validation_txt, is_test=False),
                                    batch_size=batch_size, shuffle=False)

    dataloader_dict = {'train':train_loader, 'validation':validation_loader}
    print("Total Number of Batches: ", len(train_loader))

    classifier = MyDenseNet161(num_classes=num_classes).to(device)

    print("----------"*10)
    print(classifier)
    print("----------"*10)
    centerloss = CenterLoss(num_classes, 2208).to(device)
    klloss = nn.CrossEntropyLoss()
    optimizer1 = torch.optim.SGD([{'params':classifier.features.parameters(), 'lr':0.1} , {'params':classifier.fc1.parameters(), 'lr':0.1}],
                                 lr=0.1, momentum=0.9, weight_decay=1e-4)
    optimizer2 = torch.optim.SGD(centerloss.parameters(), lr =0.5)
    scheduler1 = lr_scheduler.MultiStepLR(optimizer1, milestones=[1,2,3], gamma=0.1)
    scheduler2 = lr_scheduler.MultiStepLR(optimizer2, milestones=[4], gamma=0.1)

    train_mode = True
    if train_mode:
        # classifier.fc2 = AngleLinear(2208, 2300)
        # classifier.load_state_dict(torch.load('./Models/DenseNet_epoch24.ckpt'))
        # classifier = classifier.features
        classifier = train_classifier(classifier, klloss, centerloss, optimizer1, optimizer2, scheduler1, scheduler2)
        torch.save(classifier.state_dict(), './Models/DenseNet.ckpt')


    eval_mode = False
    if eval_mode:
        # classifier.load_state_dict(torch.load('./Models/DenseNet.ckpt'))
        test_loader = Data.DataLoader(ImageLoader("test_order_classification.txt", is_test=True),
                                    batch_size=batch_size, shuffle=False)
        final_prediciton = get_prediction(classifier, test_loader)
        write_csv(final_prediciton)
