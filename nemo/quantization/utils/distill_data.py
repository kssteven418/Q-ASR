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


def l2_loss(bn_mean, bn_std, tmp_mean, tmp_std, normalize):
    mean_loss = own_loss(bn_mean, tmp_mean, normalize)
    std_loss = own_loss(bn_std, tmp_std, normalize)
    return mean_loss + std_loss

def kl_loss(bn_mean, bn_std, tmp_mean, tmp_std):
    a = torch.log(tmp_std / bn_std)
    c = (bn_std ** 2 + (bn_mean - tmp_mean) ** 2) / tmp_std ** 2
    b = 0.5 * (1 - c)
    loss = a - b
    return loss.mean()

def hd_loss(bn_mean, bn_std, tmp_mean, tmp_std):
    v1 = bn_std ** 2
    v2 = tmp_std ** 2
    exponent = -0.25 * (bn_mean - tmp_mean) ** 2 / (v1 + v2)
    factor = (2 * bn_std * tmp_std / (v1 + v2)) ** 0.5
    loss = 1 - factor * torch.exp(exponent)
    return loss.mean()

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
                     loss_criterion='zeroq',
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

            total_loss = 0
            total_max = 0
            cnt = 0

            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for cnt, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                conv_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                conv_mean = torch.mean(conv_output[0], dim=(0, 2))
                conv_var = torch.var(conv_output[0] + eps, dim=(0, 2))
                conv_std = torch.sqrt(conv_var + eps)

                assert bn_mean.shape == conv_mean.shape
                assert bn_std.shape == conv_var.shape
                if loss_criterion in ['zeroq', 'zeroq-norm']:
                    normalize = ('norm' in loss_criterion)
                    total_loss += l2_loss(bn_mean, bn_std, conv_mean, conv_std, normalize)
                elif loss_criterion == 'kl':
                    total_loss += kl_loss(bn_mean, bn_std, conv_mean, conv_std)
                else:
                    assert loss_criterion == 'hd'
                    total_loss += hd_loss(bn_mean, bn_std, conv_mean, conv_std)

                output = conv_output[0]
                m = output.abs().max()
                '''
                with torch.no_grad():
                    p = torch.quantile(output.abs(), 0.99995)
                    panelty = m/p - 1
                #print(panelty)
                total_max += m * panelty
                '''
                total_max += m

                cnt += 1

            total_max = total_max / cnt
            #print(total_loss, total_max)
            #print(float(total_max), float(total_loss))
            total_loss += alpha * total_max

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
            if it % 1000 == 0:
                print('min, max, l2-norm:', float(gd.min()), float(gd.max()), float(l2_norm))
                print('total loss:', it, float(total_loss))
                print()

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

        refined_gaussian.append(gaussian_data.detach().clone())

    for handle in hook_handles:
        handle.remove()
    return refined_gaussian
