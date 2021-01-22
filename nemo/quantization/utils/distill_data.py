import os
import json
import torch
import torch.nn as nn
import copy
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

hard_tgt = [28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28,
        28, 20, 8, 5, 28, 0, 0, 28, 23, 8, 15, 28, 12, 5, 5, 28, 28, 28, 28, 0, 0, 0, 28, 28, 3, 28, 15, 12, 28, 
        28, 12, 28, 5, 28, 3, 28, 28, 28, 20, 9, 15, 28, 14, 28, 28, 28, 0, 0, 0, 15, 6, 0, 0, 0, 0, 0, 28, 15, 
        21, 18, 28, 0, 0, 0, 0, 0, 28, 28, 1, 16, 28, 28, 28, 28, 16, 18, 5, 28, 28, 28, 28, 3, 9, 1, 1, 28, 28, 
        28, 20, 28, 9, 22, 22, 5, 28, 28, 0, 0, 0, 0, 0, 28, 1, 20, 28, 28, 28, 20, 18, 9, 28, 28, 28, 28, 2, 28,
        21, 28, 20, 5, 5, 19, 19, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 0, 0, 0, 0,
        0, 0, 28, 8, 1, 28, 19, 28, 28, 0, 0, 0, 0, 20, 15, 0, 0, 0, 0, 0, 2, 5, 0, 0, 0, 0, 0, 0, 28, 20, 18, 5, 
        1, 1, 28, 28, 20, 28, 5, 28, 4, 28, 0, 0, 0, 0, 1, 19, 0, 0, 0, 0, 0, 0, 0, 28, 28, 6, 28, 1, 28, 12, 28, 
        12, 12, 28, 9, 9, 14, 7, 7, 0, 0, 0, 0, 0, 0, 0, 28, 15, 21, 20, 28, 28, 28, 28, 28, 28, 19, 28, 9, 28, 4, 
        5, 28, 0, 0, 0, 15, 6, 0, 0, 0, 28, 9, 20, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 
        28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 
        28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 
        28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 0, 0, 0, 0, 0, 0, 0, 0, 28, 9, 6, 28, 0, 0, 0, 0, 0, 0, 
        28, 23, 5, 28, 28, 28, 0, 0, 0, 0, 28, 28, 13, 28, 5, 1, 1, 14, 28, 28, 28, 0, 0, 0, 0, 28, 2, 28, 25, 28, 
        28, 0, 0, 0, 0, 28, 28, 28, 16, 8, 25, 28, 28, 28, 19, 28, 9, 28, 28, 28, 3, 28, 1, 12, 12, 28, 28, 28, 0, 
        0, 28, 14, 28, 1, 28, 28, 28, 28, 20, 28, 21, 28, 18, 5, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 28, 
        28, 28, 28, 28, 28, 28, 28, 28, 0, 0, 0, 0, 0, 0, 23, 8, 1, 28, 20, 28, 28, 28, 28, 5, 28, 28, 22, 28, 5, 18, 
        28, 0, 0, 0, 28, 12]

target = [23.0, 1.0, 19.0, 0.0, 5.0, 14.0, 10.0, 15.0, 25.0, 9.0, 14.0, 7.0, 0.0, 8.0, 9.0, 13.0, 19.0, 5.0, 12.0, 6.0, 0.0, 12.0, 9.0, 11.0, 5.0, 0.0, 13.0, 15.0, 19.0, 20.0, 0.0, 17.0, 21.0, 9.0, 5.0, 20.0, 0.0, 6.0, 15.0, 12.0, 11.0, 19.0, 0.0, 8.0, 5.0, 0.0, 12.0, 9.0, 11.0, 5.0, 4.0, 0.0, 20.0, 1.0, 12.0, 11.0, 1.0, 20.0, 9.0, 22.0, 5.0, 0.0, 16.0, 5.0, 15.0, 16.0, 12.0, 5.0, 0.0, 23.0, 8.0, 5.0, 14.0, 0.0, 20.0, 8.0, 5.0, 25.0, 0.0, 23.0, 5.0, 18.0, 5.0, 0.0, 23.0, 9.0, 12.0, 12.0, 9.0, 14.0, 7.0, 0.0, 20.0, 15.0, 0.0, 4.0, 15.0, 0.0]
target_len = torch.tensor([float(len(target))]).cuda() 
target = torch.tensor(target).cuda().reshape(1, -1)

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


def get_random_data(batch_size=32, dim=64, seqlen=500):
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
                     teacher_model_decoder,
                     batch_size,
                     dim,
                     seqlen,
                     train_iter=500,
                     num_batch=1,
                     alpha=0,
                     beta=0,
                     loss_fn=None,
                     ):
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
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    for conv, bn in teacher_model.convs_before_bn:
        assert isinstance(bn, nn.BatchNorm1d)
        hook = output_hook()
        hooks.append(hook)
        temp = conv.register_forward_hook(hook.hook)
        hook_handles.append(temp)

        bn_stats.append(
                (bn.running_mean.detach().clone().flatten().cuda(),
                 torch.sqrt(bn.running_var + eps).detach().clone().flatten().cuda()))

    assert len(hooks) == len(bn_stats)

    for i, gaussian_data in enumerate(dataloader):
        if i == num_batch:
            break
        print('Distillation: %s / %s' % (i+1, num_batch))
        # initialize the criterion, optimizer, and scheduler
        #gaussian_data = gaussian_data.cuda()
        #gaussian_data.requires_grad = True
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=0.05)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=True,
                                                         patience=50)
        optimizer1 = optim.Adam([gaussian_data], lr=0.05)
        scheduler1 = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=True,
                                                         patience=50)
        for it in range(train_iter):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            length = torch.tensor([seqlen] * batch_size).cuda()
            encoded, encoded_len, encoded_scaling_factor = teacher_model(gaussian_data, length) # encoder
            log_probs = teacher_model_decoder(encoder_output=encoded, encoder_output_scaling_factor=encoded_scaling_factor)
            log_probs_red = log_probs.max(axis=-1).values[-1]
            #print(log_probs_red.shape)
            print(log_probs_red)
            #print(log_probs.tolist())

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
                conv_var = torch.var(conv_output[0] + eps, dim=(0, 2))
                #print('conv mean', conv_mean)
                assert bn_mean.shape == conv_mean.shape
                assert bn_std.shape == conv_var.shape
                mean_loss += own_loss(bn_mean, conv_mean)
                std_loss += own_loss(bn_std * bn_std, conv_var)

            bn_loss = mean_loss + std_loss
            match_loss = loss_fn(log_probs=log_probs, targets=target, input_lengths=encoded_len, target_lengths=target_len)
            match_loss = match_loss / encoded_len 
            #total_loss = match_loss * 0.001 + bn_loss #TODO replace with alpha
            total_loss = bn_loss
            print('total loss:', total_loss)
            print('match loss, total loss:', match_loss, bn_loss)
            print()
            total_loss.backward()
            #print('grad', gaussian_data.grad)
            optimizer.step()
            scheduler.step(total_loss.item())
            
        print('Next step')
        for it in range(train_iter):
            teacher_model.zero_grad()
            optimizer1.zero_grad()
            for hook in hooks:
                hook.clear()
            length = torch.tensor([seqlen] * batch_size).cuda()
            encoded, encoded_len, encoded_scaling_factor = teacher_model(gaussian_data, length) # encoder
            log_probs = teacher_model_decoder(encoder_output=encoded, encoder_output_scaling_factor=encoded_scaling_factor)

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
                conv_var = torch.var(conv_output[0] + eps, dim=(0, 2))
                #print('conv mean', conv_mean)
                assert bn_mean.shape == conv_mean.shape
                assert bn_std.shape == conv_var.shape
                mean_loss += own_loss(bn_mean, conv_mean)
                std_loss += own_loss(bn_std * bn_std, conv_var)

            bn_loss = mean_loss + std_loss
            match_loss = loss_fn(log_probs=log_probs, targets=target, input_lengths=encoded_len, target_lengths=target_len) / encoded_len
            match_loss = match_loss / encoded_len 
            #total_loss = match_loss * 0.001 + bn_loss #TODO replace with alpha
            total_loss = 0.01 *  match_loss + bn_loss
            print('total loss:', total_loss)
            print('match loss, total loss:', match_loss, bn_loss)
            print()
            total_loss.backward()
            #print('grad', gaussian_data.grad)
            optimizer1.step()
            scheduler1.step(total_loss.item())
            

        refined_gaussian.append(gaussian_data.detach().clone())

    for handle in hook_handles:
        handle.remove()
    return refined_gaussian
