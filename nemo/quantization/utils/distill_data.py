import os
import json
import torch
import torch.nn as nn
import copy
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

class UniformDataset(Dataset):
    """
    Get random uniform samples from [-0.3, 0.3]
    """
    def __init__(self, length, size, transform):
        self.length = length
        self.transform = transform
        self.size = size

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = torch.rand(self.size) * 0.6 - 0.3
        return sample

class OutputHook(object):
    """
    Forward_hook used to get the output of the intermediate layer. 
    """
    def __init__(self):
        super(OutputHook, self).__init__()
        self.outputs = None

    def hook(self, module, input, output):
        self.outputs = output

    def clear(self):
        self.outputs = None

def _get_random_data(batch_size=32, dim=64, seqlen=500):
    """
    Get random sample dataloader of shape [batch_size, dim, seqlen]

    Parameters:
    ----------
    batch_size: the batch size of random data
    dim: the dimension of ramdom data
    seqlen: the sequence length of ramdom data
    """
    size = (dim, seqlen)
    dataset = UniformDataset(length=10000, size=size, transform=None)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=32)
    return data_loader

def kl_loss(bn_mean, bn_std, tmp_mean, tmp_std):
    """
    Compute KL divergence given two Gaussian distributions whose statistics
    are (bn_meanm, bn_std) and (tmp_mean, tmp_std), respectively.
    """
    a = torch.log(tmp_std / bn_std)
    c = (bn_std ** 2 + (bn_mean - tmp_mean) ** 2) / tmp_std ** 2
    b = 0.5 * (1 - c)
    loss = a - b
    return loss.mean()


def get_synthetic_data(teacher_model,
                       teacher_model_decoder,
                       batch_size,
                       dim,
                       seqlen,
                       train_iter=500,
                       num_batch=1,
                       lr=0.01,
                       ):
    """
    Generate synthetic data according to the BatchNorm statistics in the pretrained single-precision model.
    Currently only support a single GPU.

    Parameters:
    ----------
    teacher_model: encoder of the pretrained single-precision model
    teacher_model_decoder: decoder of the pretrained single-precision model
    batch_size: batch size of the synthetic data, i.e., shape[0]
    dim: dimension of the synthetic data, i.e., shape[1]
    seqlen: sequence length of the synthetic data, i.e., shape[2]
    num_batch: number of batch of the synthetic data
    lr: learning rate for the sytnthetic data generation
    """

    # initialize synthetic data with random noise according to the dataset
    dataloader = _get_random_data(batch_size, dim, seqlen)

    eps = 1e-6
    # initialize hooks and single-precision model
    teacher_model = teacher_model.cuda()
    teacher_model = teacher_model.eval()

    hooks, hook_handles, bn_stats, refined_gaussian = [], [], [], []
    for conv, bn in teacher_model.convs_before_bn:
        assert isinstance(bn, nn.BatchNorm1d)
        hook = OutputHook()
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
        gaussian_data = gaussian_data.cuda()
        gaussian_data.requires_grad = True
        crit = nn.CrossEntropyLoss().cuda()
        optimizer = optim.Adam([gaussian_data], lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         min_lr=1e-4,
                                                         verbose=True,
                                                         patience=25)
        for it in tqdm(range(train_iter)):
            teacher_model.zero_grad()
            optimizer.zero_grad()
            for hook in hooks:
                hook.clear()
            length = torch.tensor([seqlen] * batch_size).cuda()
            gd = gaussian_data
            encoded, encoded_len, encoded_scaling_factor = teacher_model(gd, length) # encoder
            log_probs = teacher_model_decoder(encoder_output=encoded, encoder_output_scaling_factor=encoded_scaling_factor)
            log_probs_red = log_probs.max(axis=-1).values[-1]
            total_loss = 0

            # compute the loss according to the BatchNorm statistics and the statistics of intermediate output
            for _, (bn_stat, hook) in enumerate(zip(bn_stats, hooks)):
                conv_output = hook.outputs
                bn_mean, bn_std = bn_stat[0], bn_stat[1]
                conv_mean = torch.mean(conv_output[0], dim=(0, 2))
                conv_var = torch.var(conv_output[0] + eps, dim=(0, 2))
                conv_std = torch.sqrt(conv_var + eps)

                assert bn_mean.shape == conv_mean.shape
                assert bn_std.shape == conv_var.shape
                total_loss += _kl_loss(bn_mean, bn_std, conv_mean, conv_std)

            total_loss.backward()
            optimizer.step()
            scheduler.step(total_loss.item())

        refined_gaussian.append(gaussian_data.detach().clone())

    for handle in hook_handles:
        handle.remove()
    return refined_gaussian
