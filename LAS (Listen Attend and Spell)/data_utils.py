import numpy as np 
import torch
import sys
import os 
import Levenshtein as Lev
import torch.utils.data as Data
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pad_sequence


def to_float_tensor(np_array):
    return torch.from_numpy(np_array).float()

def to_long_tensor(np_array):
    return torch.from_numpy(np_array).long()

def to_int_tensor(np_array):
    return torch.from_numpy(np_array).int()

def to_tensor(np_array):
    return torch.from_numpy(np_array)

def tensor_to_numpy(tensor):
    return tensor.data.cpu().numpy()

def tensor_to_variable(tensor):
    if torch.cuda.is_available():
        tensor = tensor.cuda()
    return torch.autograd.Variable(tensor)      


index2char = ['<sos>', '<eos>', ' ', "'", '+', '-', '.', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
# char2index = {'<eos>': 0, ' ': 1, "'": 2, '+': 3, '-': 4, '.': 5, 'A': 6, 'B': 7, 'C': 8, 'D': 9, 'E': 10, 'F': 11, 'G': 12, 'H': 13, 'I': 14, 'J': 15, 'K': 16, 'L': 17, 'M': 18, 'N': 19, 'O': 20, 'P': 21, 'Q': 22, 'R': 23, 'S': 24, 'T': 25, 'U': 26, 'V': 27, 'W': 28, 'X': 29, 'Y': 30, 'Z': 31, '_': 32}
char2index = {k:v for v,k in enumerate(index2char)}
if os.path.isfile('data/dictionary.npy'):
    word_dictionary = np.load('data/dictionary.npy')
    print("Dictionary loaded.")
else:
    tags = np.array(["<sos>", "<eos>"]).astype('U')
    transcripts = np.concatenate((np.load("data/train_transcripts.npy"), np.load("data/dev_transcripts.npy")))
    transcripts = np.unique(np.concatenate(transcripts))
    word_dictionary = np.concatenate((tags, transcripts))
    np.save('data/dictionary.npy', word_dictionary)
    del transcripts, tags
word2index = {k:v for v,k in enumerate(word_dictionary)}


class DataSet(Data.Dataset):
    def __init__(self, x, y, use_char=False):
        self.x = x
        self.y = y
        self.use_char = use_char

    def __len__(self):
        return self.x.shape[0]    

    def __getitem__(self, index):
        input_x = self.x[index].astype(float)
        if self.y is None:
            labels = [-1]
        else:
            if self.use_char:
                string =  " ".join([word.decode('UTF-8') for word in self.y[index]])
                y_line = list(string) + ["<eos>"]
                labels = [char2index[c] for c in y_line]
            else:
                y_line = np.concatenate((self.y[index].astype('U'), np.array(['<eos>'])))
                labels = [word2index[w] for w in y_line]

        return to_float_tensor(input_x), to_int_tensor(np.array(labels))


def collate(batch):
    '''
    outputs:
        padded_utter: a PyTorch padded tensor of the input of shape [batch_size x max_seq_len x 40]
                        with padding value 0.0. Can directly be used for conv1d.
        seq_len: a PyTorch IntTensor of shape [batch_size,], indicating the length of each input_x
        padded_label: a PyTorch LongTensor of labels, shape [batch_size x max_label_len]
        label_len: a PyTorch IntTensor of shape [batch_size,], indicating the length of each label
        pack_padded_sequece is recommend before sending to rnn.
    '''
    batch_size = len(batch)
    batch = sorted(batch, key=lambda x:x[0].size(0), reverse=True)
    max_seq_len = batch[0][0].size(0)
    freq_channels = batch[0][0].size(1)
    # [max_seq_len, batch_size, 40]
    padded_utter = torch.zeros(batch_size, max_seq_len, freq_channels)
    seq_len = torch.zeros(batch_size).int()
    max_label_len = max(labels.size(0) for (utter, labels) in batch)
    padded_labels = torch.ones(batch_size, max_label_len).long()
    label_len = torch.zeros(batch_size).int()

    for i, (utter, labels) in enumerate(batch):
        seq_len[i] = utter.size(0)
        label_len[i] = labels.size(0)
        padded_utter[i,:utter.size(0),:] = utter
        padded_labels[i,:labels.size(0)] = labels
    return padded_utter, seq_len, padded_labels, label_len


def my_DataLoader(args, use_dev_for_train=False):

    x_dev = np.load(args.data_dir + '/dev.npy', encoding='bytes')
    y_dev = np.load(args.data_dir + '/dev_transcripts.npy')

    if use_dev_for_train:
        x_train = np.load(args.data_dir + '/train.npy', encoding='bytes')
        y_train = np.load(args.data_dir + '/train_transcripts.npy')
        x_train = np.concatenate((x_train, x_dev))
        y_train = np.concatenate((y_train, y_dev))
    else:
        x_train = np.load(args.data_dir + '/train.npy', encoding='bytes')
        y_train = np.load(args.data_dir + '/train_transcripts.npy')
    print("Data Loaded")
    kwargs = {'pin_memory': True}
    train_loader = Data.DataLoader(DataSet(x_train, y_train, use_char=args.use_char), batch_size=args.batch_size,
                                                shuffle=True, collate_fn=collate, **kwargs)
    valid_loader = Data.DataLoader(DataSet(x_dev, y_dev, use_char=args.use_char), batch_size=args.batch_size,
                                                shuffle=True, collate_fn=collate, **kwargs)

    return train_loader, valid_loader


def my_testLoader(args):
    print("Loading Test Data.")
    test_x = np.load(args.data_dir + '/test.npy', encoding='bytes')
    print("Test Data loaded.")
    kwargs = {'pin_memory': True}
    test_loader = Data.DataLoader(DataSet(test_x, None), batch_size=1, shuffle=False, collate_fn=my_collate, **kwargs)
    return test_loader


def index2word(numpy_array, label_len=None):
    '''
    params:
        numpy_array: a 2d numpy int array of shape [batch_size x label_len(variable)]
    '''
    remove_list = ['<sos> ', ' <eos>']
    prediction = []
    if label_len is not None:
        for line, length in zip(numpy_array, label_len):
            text = " ".join([word_dictionary[l] for l in line[:length]])
            for tag in remove_list:
                text = text.replace(tag, '')
            prediction.append(text)
        return prediction
    else:
        for line in numpy_array:
            text = " ".join([word_dictionary[l] for l in line])
            for tag in remove_list:
                text = text.replace(tag, '')
            prediction.append(text)
        return prediction


def index2character(numpy_array):
    '''
    params:
        numpy_array: a 2d numpy int array of shape [batch_size x label_len(variable)]
    '''
    remove_list = ['<sos> ', ' <eos>']
    prediction = []
    for line in numpy_array:
        text = "".join([index2char[l] for l in line])
        for tag in remove_list:
            text = text.replace(tag, '')
        prediction.append(text)
    return prediction 


def cer(s1, s2):
    """
    Computes the Character Error Rate, defined as the edit distance.
    Arguments:
        s1 (string): space-separated sentence
        s2 (string): space-separated sentence
    """
    s1, s2, = s1.replace(' ', ''), s2.replace(' ', '')
    return Lev.distance(s1, s2)

def label_list_to_str(labels):
    output = []
    for l in labels:
        output.append("".join(index2char[i] for i in l))
    return output

def labels2str(labels, label_sizes):
    output = []
    for l, s in zip(labels, label_sizes):
        output.append("".join(index2char[i] for i in l[:s]))
    return output


if __name__ == '__main__':
    print("word vocab size:", word_dictionary.shape[0])
    print("character vocab size:", len(index2char))