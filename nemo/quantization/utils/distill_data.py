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
        sample = torch.rand(self.size) * 0.6 - 0.3
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

def own_loss(A, B, normalize):
    """
	L-2 loss between A and B normalized by length.
    Shape of A should be (features_num, ), shape of B should be (batch_size, features_num)
	"""
    if normalize:
        return (A - B).norm()**2 / (B.size(0) * A.norm()**2)
    else:
        return (A - B).norm()**2 / (B.size(0))

'''
def three_sigma_loss(bn_mean, conv_mean, bn_std, conv_std, normalize):
    A = bn_mean + 3 * bn_std
    B = conv_mean + 3 * conv_std
    if normalize:
        return (A - B).norm()**2 / (B.size(0) * A.norm()**2)
    else:
        return (A - B).norm()**2 / (B.size(0))
'''

def three_sigma_loss(m1, m2, s1, s2, output, normalize):
    #TODO: move this to a separate function
    # Hellinger distance

    v1 = s1 ** 2
    v2 = s2 ** 2
    #print(float(output.min()), float(output.mean()), float(output.max()))
    exponent = -0.25 * (m1 - m2) ** 2 / (v1 + v2)
    factor = (2 * s1 * s2 / (v1 + v2)) ** 0.5
    loss = 1 - factor * torch.exp(exponent)
    l2 = ((output ** 2).mean(axis=0).mean(axis=-1) + 1e-6).sqrt()
    #print(loss.sum(), l2.sum())
    return loss.sum(), l2.sum(), len(loss)

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
                     normalize=True,
                     three_sigma=False,
                     lr=0.01,
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
        print('Distillation: %s / %s' % (i+1, num_batch), alpha, lr)
        # initialize the criterion, optimizer, and scheduler
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=True,
                                                         patience=25)
        for it in range(train_iter):
            '''
            # Uncomment this for step-by-step data distillation analysis
            refined_gaussian.append(gaussian_data.detach().clone())
            '''

            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            length = torch.tensor([seqlen] * batch_size).cuda()
            gd = gaussian_data
            '''
            # Uncomment this for data normalization and clipping before the encoder layer
            v = gd.std(axis=-1, keepdim=True)
            m = gd.mean(axis=-1, keepdim=True)
            gd = (gd - m) / v
            gd = torch.clamp(gd, min=-4., max=4.)
            '''
            encoded, encoded_len, encoded_scaling_factor = teacher_model(gd, length) # encoder
            log_probs = teacher_model_decoder(encoder_output=encoded, encoder_output_scaling_factor=encoded_scaling_factor)
            log_probs_red = log_probs.max(axis=-1).values[-1]

            mean_loss = 0
            std_loss = 0
            length = 0

            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                conv_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                conv_mean = torch.mean(conv_output[0], dim=(0, 2))
                conv_var = torch.var(conv_output[0] + eps, dim=(0, 2))
                assert bn_mean.shape == conv_mean.shape
                assert bn_std.shape == conv_var.shape
                if not three_sigma:
                    mean_loss_ = own_loss(bn_mean, conv_mean, normalize=normalize)
                    std_loss_ = own_loss(bn_std * bn_std, conv_var, normalize=normalize)
                    mean_loss += mean_loss_ 
                    std_loss += std_loss_
                else:
                    conv_std = (conv_var + eps) ** 0.5 
                    mean_loss_, l2_, len_ = three_sigma_loss(bn_mean, conv_mean, bn_std, conv_std, conv_output[0], normalize=normalize)
                    mean_loss += mean_loss_ + alpha * l2_
                    length += len_
                #print(cnt)
                #print(float(bn_mean.abs().max()), float(bn_mean.abs().mean()), float(mean_loss_))
                #print(float(bn_std.max()), float(bn_std.mean()), float(std_loss_))
                #print()

            bn_loss = mean_loss + std_loss
            if length != 0:
                bn_loss = bn_loss / length
            total_loss = bn_loss

            '''
            # Uncomment this for regularization
            x = (gd[:, :, 1:] - gd[:, :, :-1]).abs()
            x = (x[:, 1:, :] - x[:, :-1, :]).abs()
            tv_grad = x.mean()
            total_loss += alpha * tv_grad # by default, alpha==0

            log_prob_loss = log_probs_red.mean().abs()
            total_loss += beta * log_prob_loss
            '''

            # Uncomment this for logging
            l2_norm = torch.sqrt(gd * gd).mean()
            if it % 1 == 0:
                print(float(gd.min()), float(gd.max()), float(l2_norm))
                #print('TV gradient regularization', float(tv_grad))
                #print('Log prob mean', float(log_prob_loss))
                #print('bn_loss', float(bn_loss))
                print('total loss:', it, float(total_loss))
                print()

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

        refined_gaussian.append(gaussian_data.detach().clone())

    for handle in hook_handles:
        handle.remove()
    return refined_gaussian
