import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Function, Variable
from torch.nn.parameter import Parameter
import math
import torch
import torch.nn as nn
from torch.autograd.function import Function


class CenterLoss(nn.Module):
    '''
    References.
    https://github.com/jxgu1016/MNIST_center_loss_pytorch
    '''
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, feature, label):
        batch_size = feature.size(0)
        feature = feature.view(batch_size, -1)

        if feature.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feature.size(1)))
        batch_size_tensor = feature.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feature, label, self.centers, batch_size_tensor)
        # print(loss)
        # print(type(loss))
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(cls, feature, label, centers, batch_size_tensor):
        cls.save_for_backward(feature, label, centers, batch_size_tensor)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size_tensor

    @staticmethod
    def backward(cls, grad_output):
        feature, label, centers, batch_size_tensor = cls.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        div = centers_batch - feature

        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), div)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * div / batch_size_tensor, None, grad_centers / batch_size_tensor, None


class AgentCenterLoss(nn.Module):

    def __init__(self, num_classes, input_dim, scale):
        super(AgentCenterLoss, self).__init__()
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.scale = scale

        self.centers = nn.Parameter(torch.randn(self.num_classes, self.input_dim))

    def forward(self, x, labels):

        cos_dis = Flinear(F.normalize(x), F.normalize(self.centers)) * self.scale

        one_hot = torch.zeros_like(cos_dis)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # loss = 1 - cosine(i)
        loss = one_hot * self.scale - (one_hot * cos_dis)

        return loss.mean()


def myphi(x,m):
    x = x * m
    return 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6) + \
            x**8/math.factorial(8) - x**9/math.factorial(9)

class AngleLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 4, phiflag=True):
        super(AngleLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.mlambda = [
            lambda x: x**0, # cos(0)
            lambda x: x**1, # cos(theta)
            lambda x: 2*x**2-1, # cos(2*theta)
            lambda x: 4*x**3-3*x,
            lambda x: 8*x**4-8*x**2+1,
            lambda x: 16*x**5-20*x**3+5*x
        ]

    def forward(self, inputs):
        x = inputs   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features

        ww = w.renorm(2,1,1e-5).mul(1e5)
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = ww.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(ww) # size=(B,Classnum)
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)
        cos_theta = cos_theta.clamp(-1,1)

        if self.phiflag:
            cos_m_theta = self.mlambda[self.m](cos_theta)
            theta = Variable(cos_theta.data.acos())
            k = (self.m*theta/3.14159265).floor()
            n_one = k*0.0 - 1
            phi_theta = (n_one**k) * cos_m_theta - 2*k
        else:
            theta = cos_theta.acos()
            phi_theta = myphi(theta,self.m)
            phi_theta = phi_theta.clamp(-1*self.m,1)

        cos_theta = cos_theta * xlen.view(-1,1)
        phi_theta = phi_theta * xlen.view(-1,1)
        output = (cos_theta, phi_theta)
        return output # size=(B,Classnum,2)


class AngleLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AngleLoss, self).__init__()
        self.gamma = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi_theta = input
        target = target.view(-1,1) #size=(B,1)

        index = cos_theta.data * 0.0 #size=(B,Classnum)
        index.scatter_(1,target.data.view(-1,1),1)
        index = index.byte()
        index = Variable(index)

        self.lamb = max(self.LambdaMin,self.LambdaMax/(1+0.1*self.it ))
        output = cos_theta * 1.0 #size=(B,Classnum)
        output[index] -= cos_theta[index]*(1.0+0)/(1+self.lamb)
        output[index] += phi_theta[index]*(1.0+0)/(1+self.lamb)

        logpt = F.log_softmax(output)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss.mean()
        return loss