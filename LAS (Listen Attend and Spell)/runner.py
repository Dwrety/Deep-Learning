import argparse
import time
from datetime import datetime
import os 
import csv 
from datetime import datetime
import math
import numpy as np
from data_utils import *
from modules import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


parser = argparse.ArgumentParser(description="Pytorch ASR implementation of LAS")
parser.add_argument('-data_dir', type=str, default='./data', help='directory to wsj data corpora.')
parser.add_argument('-use_char', type=bool, default=True, help='whether to use char or word.')
parser.add_argument('-weights_dir', type=str, default='./weights/', help='data directory')
parser.add_argument('-lr', type=float, default=1e-4, help='initial learning rate')
parser.add_argument('-clip_norm', type=float, default=1, help='clip the gradient larger than this value')
parser.add_argument('-epochs', type=int, default=100, help='number of epochs')
parser.add_argument('-batch_size', type=int, default=32, metavar='N', help='batch size')
parser.add_argument('-seed', type=int, default=9999, help='environmental seed for stochastic events')
parser.add_argument('-cuda', action='store_false', help='use CUDA GPU')
parser.add_argument('-log', type=int, default=20, metavar='N', help='time interval between each log')
parser.add_argument('-weight_decay', type=float, default=1e-6, help='weight decay applied to all weights')

args = parser.parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def find_first_eos(pred):
    pred: [label_len, vocab_size]
    labels = pred.max(1)[1].data.cpu().numpy()
    for idx, label in enumerate(labels):
        if label == char2index['<eos>']:
            return idx
    return len(labels)


def greedy_search(preds):
    # probs: FloatTensor (B, L, C)
    out = []
    for pred in preds:
        s = []
        for step in pred:
            #             idx = torch.multinomial(step, 1)[0]
            idx = step.max(0)[1].item()
            c = index2char[idx]
            s.append(c)
            if c == '<eos>':
                break
        out.append("".join(s))
    return out

def greedy_search_word(preds):
    out = []
    for pred in preds:
        s = []
        for step in pred:
            idx = step.max(0)[1].item()
            w = word_dictionary[idx]
            if w == '<eos>':
                break
            s.append(w)
        out.append(" ".join(s))
    return out

def random_search_decoder(model, input_x, seq_len, N=5, max_iter=250):
    def generator_len(gen):
        for idx, c in enumerate(gen):
            if c == char2index['<eos>']:
                return idx + 1
        return len(gen)

    random_gens = []
    criterion = Seq_CrossEntropy()
    for n in range(N):
        state, pred = model.get_init_state(input_x, seq_len)
        state = [state]
        pred = [pred]
        gen = []
        gen_probs = []
        for i in range(max_iter):
            state, probs, attention_score = model.generate(input_x, pred, state)
            ci = torch.multinomial(to_tensor(probs[0]), 1).squeeze().numpy().item()
            pred = [ci]
            gen.append(ci)
            gen_probs += probs
            if ci == char2index['<eos>']:
                break

        all_labels = torch.zeros(1, len(gen)).long()
        for i, l in enumerate(gen):
            all_labels[0,i] = int(l)
        # print(all_labels.size())
        # print(Variable(to_tensor(np.array(gen_probs)).unsqueeze(0)).size())

        perplexity = criterion(Variable(to_tensor(np.array(gen_probs)).unsqueeze(0)),
                               torch.from_numpy(np.array([generator_len(gen)])),
                                Variable(all_labels))[1]
        random_gens.append((gen, perplexity))
    random_gens.sort(key=lambda x : x[1])
    # print(random_gens[0][0])
    return label_list_to_str([random_gens[0][0]])


def beam_search_decoder(preds, K):
    top_k = [[[], 1]]
    for pred in preds:
        for step in pred:
            all_candidates = list()
            for i in range(len(top_k)):
                seq, score = top_k[i]
                for j in range(len(step)):
                    candidate = [seq + [j], score * -log(step[j])]
                    all_candidates.append(candidate)
            ordered = sorted(all_candidates, key=lambda tup:tup[1])
            sequences = ordered[:K]
    return top_k


class Seq_CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, preds, label_len, golden):
        '''
        preds: [batch_size, max_iter/max_label_len, vocab_size]
        label_len: [batch_size, ]
        golden: [batch_size, max_label_len]
        to generate loss/label and perplexity
        '''
        # pred_list = []
        # label_list =  []
        # num_labels = 0
        # max_iter = preds.size(1)
        # # print(max_iter)

        # for (pred, label, num_iter) in zip(preds, golden, label_len.cpu().numpy()):
        #     pred_for_loss = []
        #     label_for_loss = []
            
        #     eos_idx = find_first_eos(pred)
        #     # print(eos_idx*32)
        #     if eos_idx < num_iter:
        #         if eos_idx != 0:
        #             pred_for_loss.append(pred[:eos_idx])
        #         pred_for_loss += [pred[eos_idx:eos_idx+1]] * (num_iter - eos_idx)
        #         label_for_loss.append(label[:num_iter])
        #         num_labels += num_iter
        #     elif eos_idx == max_iter:
        #         pred_for_loss.append(pred[:eos_idx])
        #         label_for_loss.append(label[:num_iter])
        #         label_for_loss += [label[num_iter-1:num_iter]] * (eos_idx - num_iter)
        #         # label_for_loss.append(label)
        #         num_labels += max_iter
        #     else:
        #         pred_for_loss.append(pred[:eos_idx + 1])
        #         label_for_loss.append(label[:num_iter])
        #         label_for_loss += [label[num_iter - 1:num_iter]] * (eos_idx + 1 - num_iter)
        #         num_labels += eos_idx + 1 

        #     pred_list.append(torch.cat(pred_for_loss))
        #     label_list.append(torch.cat(label_for_loss))

        # preds_batch = torch.cat(pred_list)
        # labels_batch = torch.cat(label_list)
        # # print(len(labels_batch))
        # loss = torch.nn.functional.cross_entropy(preds_batch, labels_batch, reduction='sum')
        # # del preds_batch, labels_batch
        # perplexity = np.exp(loss.item()/len(labels_batch))



        # first make them equally shaped to [batch_size, num_classes, max_label_len]
        # golden: [batch_size, max_label_len]
        # print(preds.size())
        preds = preds.permute(0, 2, 1)

        num_labels = sum(label_len).item()
        if preds.size(2) > golden.size(1):
            preds_ = preds[:, :, :golden.size(1)]
        else:
            preds_ = preds
        # print(preds_.shape, golden.shape, label_len.shape)
        max_iter = preds_.size(1)
        num_utters = preds_.size(0)

        loss = nn.functional.cross_entropy(preds_, golden, reduction='none')
        mask = Variable(loss.data.new(loss.size(0), loss.size(1)).zero_(), requires_grad=False)
        # loss [batch_size, max_label_len]
        for i, length in enumerate(label_len):
            mask[i, :length] = 1
        # print(loss)
        loss = (loss * mask).sum()

        loss_per_label = loss.item() / num_labels
        loss = loss / golden.size(0)
        return loss, loss_per_label


def plot_grad_flow(named_parameters, epoch):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        # print(n, p)
        if (hasattr(p.grad, "data")) and ("bias" not in n):
            # print(p.grad)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.03) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("epoch {}.png".format(epoch), bbox_inches='tight')
    plt.close()


def train(epoch, model, optimizer, criterion, loader, args, teacher_force):
    # Turn on training mode which enables dropout.
    model.train()
    sum_loss, num_utters, sum_labels, perplexity = 0, 0, 0, 0
    start_time = time.time()
    optimizer.zero_grad()

    # for batch, (frames, seq_sizes, labels, label_sizes) in enumerate(loader):
    for batch_id, (input_x, seq_len, golden, label_len) in enumerate(loader):
        batch_id += 1

        # frames: (max_seq_len, batch_size, channels)
        # labels: (batch, individual_label_len)
        #         print("frames.size()", frames.size())
        #         print("labels.size()", labels.size())


        sum_labels += 1
        num_utters += 1
        # sum_labels += sum(label_len).item()

        input_x, golden = input_x.to(device), golden.to(device)
        input_x, golden = Variable(input_x), Variable(golden)
        label_len = label_len.to(device)

        output, attention_score = model(input_x, seq_len, golden, teacher_force)
        # print(output.shape, attention_score.shape)

        loss, loss_per_label = criterion(output, label_len, golden)
        loss.backward()
        # plot_grad_flow(model.named_parameters(), 1)    
        sum_loss += loss.item()
        perplexity += loss_per_label
        # print(loss.item())


        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)

        if batch_id % args.log == 0:
            plot_grad_flow(model.named_parameters(), epoch)

        optimizer.step()
        optimizer.zero_grad()

        if batch_id % args.log == 0:
            plot_attention(attention_score, 0, epoch)

            #             decoded = decode(output, seq_sizes)
            #             label_str = [label2str(label) for label in labels]
            #             for l, m in zip(label_str, decoded):
            #                 print("Label:\t{}\nMine:\t{}\n".format(l, m))
            #                 break
            elapsed = time.time() - start_time
            avg_loss = sum_loss / num_utters
            perplexity = np.exp(perplexity / sum_labels)
            print(
                '| epoch {:2d} | {:4d}/{:1d} batches ({:5.2f}%) | lr {:.2e} | {:3.0f} ms/utter | loss/utter {:5.2f} | perplexity {:5.2f} |'
                    .format(epoch, batch_id, len(loader), (100.0 * batch_id / len(loader)),
                            optimizer.param_groups[0]['lr'],
                            elapsed * 1000.0 / (args.log * args.batch_size),
                            avg_loss,
                            perplexity))
            sum_loss, num_utters, sum_labels, perplexity = 0, 0, 0, 0
            start_time = time.time()


def print_score_board(score_board, N=args.batch_size):
    # sort by err asc
    score_board.sort(key=lambda x: x[0])
    # print("\nTop-{}\n".format(N))
    # print(len(score_board))
    print("\n".join(["Sentence {}, CER:\t{:6.4f}\nGolden:\t{}\nPred:\t{}\n".format(i[0], i[1], i[2], i[3]) for i in score_board]))
    # print("\nLast-{}\n".format(N))
    # print("\n".join(["CER:\t{:6.4f}\nGolden:\t{}\nPred:\t{}\n".format(i[0], i[1], i[2]) for i in score_board[-N:]]))


def evaluate(model, epoch, criterion, loader, args, calc_cer=False):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    total_cer = 0
    sum_labels = 0
    perplexity = 0
    num_utters = 0

    score_board = []
    for batch_id, (input_x, seq_len, golden, label_len) in enumerate(loader):
        batch_id += 1
        # sum_labels += sum(label_len).cpu().numpy()
        input_x, golden = input_x.to(device), golden.to(device)
        with torch.no_grad():
            input_x, golden = Variable(input_x), Variable(golden)
            label_len = label_len.to(device)
            # sum_labels += sum(label_len).item()
            num_utters += 1
            sum_labels += 1
            output, attention_score = model(input_x, seq_len, golden)

            loss, loss_per_label = criterion(output, label_len, golden)
            total_loss += loss.item()
            perplexity += loss_per_label

            if (batch_id == 1) or calc_cer:
                attention_plots = np.random.choice(args.batch_size, 1, replace=False)
                output = torch.nn.functional.softmax(output, dim=-1).data.cpu()
                decoded = greedy_search(output)

                labels_str = labels2str(tensor_to_numpy(golden), label_len)
                # labels_str = index2word(tensor_to_numpy(golden), label_len=label_len)  # [label2str(label) for label in tensor_to_numpy(labels)]
                g = 0
                for l, m in zip(labels_str, decoded):
                    e = cer(l, m)
                    total_cer += e
                    score_board.append([g, e, l, m])
                    if g in attention_plots:
                        plot_attention(attention_score, g, epoch)
                    g += 1
    perplexity = perplexity / sum_labels
    print_score_board(score_board)
    return total_loss/num_utters, np.exp(perplexity), total_cer / 1106


def plot_attention(attention_score, idx=0, epoch=1):
    # print(attention_score.shape)
    fig = plt.figure()
    plt.imshow(tensor_to_numpy(attention_score[idx]), cmap='gray')
    fig.savefig("epoch-{}-sentence{}_attention.png".format(epoch, idx))
    plt.close()


def main(args):
    print(args)

    if args.use_char:
        vocab_size = len(index2char)
    else:
        vocab_size = len(word_dictionary)
    train_loader, valid_loader = my_DataLoader(args, use_dev_for_train=False)
    model = LAS(vocab_size, pre_train=False).to(device)
    print(model)
    # for name, params in model.named_parameters():
        # print(name)
    # model.load_state_dict(torch.load("weights/"))
    model.load_state_dict(torch.load("weights/051_8.7586_pretrain.weights"))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = Seq_CrossEntropy()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, threshold=0.01, verbose=True, min_lr=1e-7)
    teacher_force = 0.1
    for epoch in range(39, args.epochs + 1):

        print(datetime.now())
        epoch_start_time = time.time()
        # if epoch >= 8:
        #     teacher_force = 0.5
        # elif epoch >= 5:
        #     teacher_force = 0.75
        # elif epoch >= 4:
        #     teacher_force = 0.8
        # elif epoch >= 2:
        #     teacher_force = 0.9

        train(epoch, model, optimizer, criterion, train_loader, args, teacher_force)
        # plot_grad_flow(model.named_parameters(), epoch)

        # validate 
        val_loss, perplexity, val_cer = evaluate(model, epoch, criterion, valid_loader, args, calc_cer=True)
        scheduler.step(val_loss)
        print('-' * 100)
        print('| Epoch {:3d} results | time: {:5.2f}s | valid loss {:5.4f} | perplexity {:5.4f} | valid cer {:5.4f} |'
              .format(epoch, (time.time() - epoch_start_time),
                      val_loss,perplexity, val_cer))
        print('-' * 100)

        if not os.path.exists(args.weights_dir):
            os.makedirs(args.weights_dir)
        weight_fname = "{}/{:03d}_{}_pretrain.weights".format(args.weights_dir, epoch, "{:.4f}".format(val_cer))
        print("saving as", weight_fname)
        torch.save(model.state_dict(), weight_fname)


def get_test_loader(args):
    print(args.data_dir)
    x_test = np.load('data/test.npy', encoding='bytes')
    test_loader = Data.DataLoader(DataSet(x_test, None, use_char=args.use_char), batch_size=1, shuffle=False, collate_fn=collate)
    return test_loader



class BeamSearchDecoder(object):
    """Beam Search decoder."""

    def __init__(self, model, beam_size, input_x, seq_len):

        # self.generation = []
        self.model = model
        self.model.eval()
        self.max_iters = 250
        self.beam_size = beam_size
        # self.current_state = []
        self.keep_k = 15
        initial_state, sos_seq = self.get_init_state()

        self.accum_states = [[[sos_seq], [initial_state], (0, 0)]]
        # [[[yt_vector], [ht vector], (score, length)],      
        #   ........               ]
        self.hs, self.state_len = self.get_src_states(input_x, seq_len)


    def get_src_states(self, input_x, seq_len):
        hs, state_len = self.model.listener(input_x, seq_len)
        return hs, state_len

    def get_init_state(self):
        initial_state, sos_seq = self.model.speller.get_init_state(batch_size=1)
        return initial_state, sos_seq

    def decode_one_step(self, hs, state_len, yt, ht):
        # print(hs.shape, state_len, yt)

        current_yt, hidden_state, alpha = self.model.speller.predict_one(hs, state_len, yt, ht, dropout_masks=None)
        # self.current_state.append(yt)
        return current_yt.squeeze(), hidden_state
        
    def length_normalize(self, length, alpha=0.7):
        return np.power((5+length)/6, alpha)

    def advance(self):
        
        num_eos = 0
        with torch.no_grad():
            for step in range(self.max_iters):
                # print(step)
                if num_eos == self.keep_k:
                    break
                else:
                    candidate_seq = []
                    current_states = [(seq[0][-1], seq[1][-1], seq[2]) for seq in self.accum_states]

                    for number, state in enumerate(current_states):
                        yt, ht, (score, length) = state
                        parent = self.accum_states[number]
                        if yt.item() == char2index['<eos>']:
                            num_eos += 1
                            candidate_seq.append(parent)
                            continue

                        new_yt_vector, new_hidden = self.decode_one_step(self.hs, self.state_len, yt, ht)
                        # print(yt)
                        new_yt_vector = nn.functional.softmax(new_yt_vector, dim=-1)

                        top_k = torch.topk(new_yt_vector, self.beam_size)
                        length += 1
                        lp = self.length_normalize(length)
                        # print(parent[0] + [1], parent[1])
                        for prob, label in zip(top_k[0], top_k[1]):
                            new_score = score - torch.log(prob)/lp
                            # print(label, new_hidden)
                            # print(label.unsqueeze(0))
                            candidate_seq.append([parent[0] + [label.unsqueeze(0)], parent[1] + [new_hidden], (new_score, length)])

                    candidate_seq = sorted(candidate_seq, key=lambda x:x[2][0])[:self.keep_k]
                    # print(candidate_seq)
                    self.accum_states = candidate_seq
        # print(self.accum_states[0])
        return self.accum_states[0]

    def translate(self, states):
        # print(index2char[states[0][0].item()])
        text = "".join([index2char[label.item()] for label in states[0]])
        text = text.replace('<eos>', '')
        text = text.replace('<sos>', '')
        return text



def predict(args):
    import csv 
    if args.use_char:
        vocab_size = len(index2char)
    else:
        vocab_size = len(word_dictionary)

    test_loader = get_test_loader(args)
    model = LAS(vocab_size).to(device)
    model.load_state_dict(torch.load("weights/051_8.7586_pretrain.weights"))
    model.eval()

    with open("submission.csv", 'w', newline='') as f:
        wrt = csv.DictWriter(f, fieldnames=['Id', 'Predicted'])
        wrt.writeheader()
        for idx, (instance, instance_len, labels, label_len) in enumerate(test_loader):
            instance = instance.to(device)
            instance_len = instance_len.to(device)
            instance = Variable(instance, requires_grad=False)

            output, attention_score = model(instance, instance_len, None, max_iters=250)
            output = nn.functional.softmax(output.squeeze(), dim=-1).unsqueeze(0).data.cpu()
            decoded = greedy_search(output)
            # decoded = random_search_decoder(model, instance, instance_len, N=10)
            for p in decoded:
                p = p.replace("<eos>", "")
                p = p.replace("<sos>", "")
                print(idx/len(test_loader)*100, "% Pred: " + p + "\n")
                wrt.writerow({'Id': idx, "Predicted": p})
        print("finished")


def predict_beam(args):
    import csv 
    if args.use_char:
        vocab_size = len(index2char)
    else:
        vocab_size = len(word_dictionary)

    test_loader = get_test_loader(args)
    model = LAS(vocab_size).to(device)
    model.load_state_dict(torch.load("weights/036_7.8237_pretrain.weights"))
    model.eval()

    with open("submission.csv", 'w', newline='') as f:
        wrt = csv.DictWriter(f, fieldnames=['Id', 'Predicted'])
        wrt.writeheader()
        for idx, (instance, instance_len, labels, label_len) in enumerate(test_loader):
            instance = instance.to(device)
            instance_len = instance_len.to(device)
            instance = Variable(instance, requires_grad=False)

            decoder = BeamSearchDecoder(model, 15, instance, instance_len)
            states = decoder.advance()
            decoded = decoder.translate(states)
            print(idx/len(test_loader)*100, "% Pred: " + decoded + "\n")
            wrt.writerow({'Id': idx, "Predicted": decoded})
        print("finished")



if __name__ == "__main__":

    # main(args)
    # predict(args)
    predict_beam(args)
    # if args.use_char:
    #     vocab_size = len(index2char)
    # else:
    #     vocab_size = len(word_dictionary)
    # train_loader, valid_loader = my_DataLoader(args, use_dev_for_train=True)
    # model = LAS(vocab_size).to(device)
    # model.load_state_dict(torch.load("weights/007_33.0938.weights"))
    # criterion = Seq_CrossEntropy()
    # val_loss_label, perplexity, val_cer = evaluate(model, 1, criterion, valid_loader, args, calc_cer=False)




