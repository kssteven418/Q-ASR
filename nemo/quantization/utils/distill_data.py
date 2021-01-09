import os
import json
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class UniformDataset(Dataset):
    """
    get random uniform samples with mean 0 and variance 1
    """
    def __init__(self, length, size, transform):
        self.length = length
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # var[U(-128, 127)] = (127 - (-128))**2 / 12 = 5418.75
        sample = torch.rand(self.size) * 10. - 5.
        return sample


def get_random_data(batch_size=32, dim=64, seqlen=700):
    """
    get random sample dataloader 
    dataset: name of the dataset 
    batch_size: the batch size of random data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """
    size = (dim, seqlen)
    dataset = UniformDataset(length=10000, size=size, transform=None)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32)
    return data_loader

def own_loss(A, B):
    """
	L-2 loss between A and B normalized by length.
    Shape of A should be (features_num, ), shape of B should be (batch_size, features_num)
	"""
    return (A - B).norm()**2 / B.size(0)

class output_hook(object):
    """
	Forward_hook used to get the output of the intermediate layer. 
	"""
    def __init__(self):
        super(output_hook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None


class ModelWrapper(torch.nn.Module):
    def __init__(self, model, length):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.length = length
        self.convs_before_bn = model.convs_before_bn

    def forward(self, x):
        return self.model(x, length=self.length)


def get_distill_data(teacher_model,
                     batch_size,
                     dim,
                     seqlen,
                     num_batch=1):
    """
    Generate distilled data according to the BatchNorm statistics in the pretrained single-precision model.
    Currently only support a single GPU.

    teacher_model: pretrained single-precision model
    dataset: the name of the dataset
    batch_size: the batch size of generated distilled data
    num_batch: the number of batch of generated distilled data
    for_inception: whether the data is for Inception because inception has input size 299 rather than 224
    """

    # initialize distilled data with random noise according to the dataset
    dataloader = get_random_data(batch_size, dim, seqlen)

    eps = 1e-6
    # initialize hooks and single-precision model
    length = torch.tensor([seqlen] * batch_size, dtype=torch.int64).cuda()
    teacher_model = ModelWrapper(teacher_model, length)
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    for m in teacher_model.convs_before_bn:
        hook = output_hook()
        hooks.append(hook)
        temp = m.register_forward_hook(hook.hook)
        hook_handles.append(temp)

    for n, m in teacher_model.named_modules():
        if isinstance(m, nn.BatchNorm1d):
            # get the statistics in the BatchNorm layers
            bn_stats.append(
                (m.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(m.running_var + eps).detach().clone().flatten().cuda()))

    assert len(hooks) == len(bn_stats)

    for i, gaussian_data in enumerate(dataloader):
        print(i)
        if i == num_batch:
            break
        # initialize the criterion, optimizer, and scheduler
        #gaussian_data = gaussian_data.cuda()
        #gaussian_data.requires_grad = True
        gaussian_data = Variable(gaussian_data.cuda(), requires_grad=True)
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=False,
                                                         patience=100)
        for it in range(500):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            output = teacher_model(gaussian_data)
            print(gaussian_data)
            print(length)
            print('*' * 50)
            print(output)
            #mean_loss = Variable(torch.zeros(1).cuda(), requires_grad=True)
            #std_loss = Variable(torch.zeros(1).cuda(), requires_grad=True)
            mean_loss = 0
            std_loss = 0

            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                #print(mean_loss)
                conv_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                conv_mean = torch.mean(conv_output[0], dim=(0, 2))
                conv_std = torch.sqrt(torch.var(conv_output[0] + eps, dim=(0, 2)))
                #print('conv mean', conv_mean)
                assert bn_mean.shape == conv_mean.shape
                assert bn_std.shape == conv_std.shape
                mean_loss += own_loss(bn_mean, conv_mean)
                std_loss += own_loss(bn_std, conv_std)

                '''
                tmp_mean = torch.mean(tmp_output.view(tmp_output.size(0),
                                                      tmp_output.size(1), -1), dim=2)
                tmp_std = torch.sqrt(
                    torch.var(tmp_output.view(tmp_output.size(0),
                                              tmp_output.size(1), -1),
                              dim=2) + eps)
                mean_loss += own_loss(bn_mean, tmp_mean)
                std_loss += own_loss(bn_std, tmp_std)
                '''
            total_loss = mean_loss + std_loss
            print(total_loss)
            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())
