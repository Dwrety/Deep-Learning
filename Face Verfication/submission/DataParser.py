import numpy as np 
import matplotlib.pyplot as plt 
import os 
import sys

level = 1
train_level = {0:'medium', 1:'large'}
train_directory = "./train_data/{}".format(train_level[level])
validation_directory = "./validation_classification/{}".format(train_level[level])

large_train_list = ["./train_data/{}".format(train_level[0]), "./train_data/{}".format(train_level[1])]
large_validation_list = ["./validation_classification/{}".format(train_level[0]), "./validation_classification/{}".format(train_level[1])]


all_classes = os.listdir(train_directory)
num_classes = len(all_classes)
print(num_classes)


if level == 1:
    # all_classes = []
    num_classes = 0
    with open('train_order_classification_{}.txt'.format(train_level[level]),'w') as f:
        for directory in large_train_list:
            all_classes = os.listdir(directory)
            num_classes += len(all_classes)
            for y in all_classes:
                for image in os.listdir(os.path.join(directory, y)):
                    if not image.startswith('.'):
                    # print(os.path.join(train_directory, y, image))
                        f.write('{};{}\n'.format(os.path.join(directory, y, image), y))

    with open('validation_order_classification_{}.txt'.format(train_level[level]),'w') as f:                
        for directory in large_validation_list:
            all_classes = os.listdir(directory)
            num_classes += len(all_classes)
            for y in all_classes:
                for image in os.listdir(os.path.join(directory, y)):
                    if not image.startswith('.'):
                    # print(os.path.join(train_directory, y, image))
                        f.write('{};{}\n'.format(os.path.join(directory, y, image), y))



# with open('train_order_classification_{}.txt'.format(train_level[level]),'w') as f:
#     for y in all_classes:
#         print(y)
#         for image in os.listdir(os.path.join(train_directory, y)):
#             if not image.startswith('.'):
#                 # print(os.path.join(train_directory, y, image))
#                 f.write('{};{}\n'.format(os.path.join(train_directory, y, image), y))

# with open('validation_order_classification_{}.txt'.format(train_level[level]),'w') as f:
#     for y in all_classes:
#         print(y)
#         for image in os.listdir(os.path.join(validation_directory, y)):
#             if not image.startswith('.'):
#                 # print(os.path.join(train_directory, y, image))
#                 f.write('{};{}\n'.format(os.path.join(validation_directory, y, image), y))
