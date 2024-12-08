# @title CL-BRUNO for instance, label and task incremental learning
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.nn.utils.parametrizations import weight_norm


def logit_transformation(x):
    # x = N*C*H*W, map from 0,1 to R
    # output y = N*C*H*W, jac = N
    alpha = 1e-3
    y = x * (1 - alpha) + alpha * 0.5
    jac = -torch.log(y) - torch.log(1 - y)
    y = torch.log(y) - torch.log(1. - y)
    return y, jac.sum(dim=[1, 2, 3])


def dequantization_forward_and_jacobian(x):
    # x = N*C*H*W, map from 0,256 to 0,1
    # output y = N*C*H*W, jac = N
    y = x / 256.0
    jac = -1. * torch.log(torch.tensor(256.0, dtype=torch.float, device=x.device)) * x.shape[1] * x.shape[2] * x.shape[3]
    return y, jac * torch.ones(x.shape[0], device=jac.device)


class MyConv(nn.Module):
    # a conditional conv layer for img
    # input = N*c_in*h*w Output = N*n_filter*h*w
    def __init__(self, c_in, cond_dim, activation=None, n_filter=64, kernel_size=3, n_hidden=128, layer=1, wn=False):
        super().__init__()
        self.cond_dim = cond_dim
        self.c_in = c_in
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.activation = activation
        self.cond_model = None
        if wn:
            if cond_dim is not None:
                cond_model = [weight_norm(nn.Linear(cond_dim, n_hidden), name='weight')]
                for _ in range(layer - 1):
                    cond_model += [nn.ReLU(), weight_norm(nn.Linear(n_hidden, n_hidden), name='weight')]
                cond_model += [nn.ReLU(), weight_norm(nn.Linear(n_hidden, n_filter), name='weight')]
                self.cond_model = nn.Sequential(*cond_model)

            conv_model = [weight_norm(nn.Conv2d(in_channels=c_in, out_channels=n_filter, kernel_size=kernel_size,
                                      stride=1, padding='same'), name='weight')]
            self.conv_model = nn.Sequential(*conv_model)
        else:
            if cond_dim is not None:
                cond_model = [nn.Linear(cond_dim, n_hidden)]
                for _ in range(layer-1):
                    cond_model += [nn.ReLU(), nn.Linear(n_hidden, n_hidden)]
                cond_model += [nn.ReLU(), nn.Linear(n_hidden, n_filter)]
                self.cond_model = nn.Sequential(*cond_model)

            conv_model = [nn.Conv2d(in_channels=c_in, out_channels=n_filter, kernel_size=kernel_size,
                                    stride=1, padding='same')]
            self.conv_model = nn.Sequential(*conv_model)

    def forward(self, x, cond_emb):
        # if self.activation is None:
        #     return self.conv_model(x) + self.cond_model(cond_emb)[:, :, None, None]
        # else:
        #     return self.activation(self.conv_model(x) + self.cond_model(cond_emb)[:, :, None, None])
        if self.cond_model is not None:
            return self.conv_model(x) + self.cond_model(cond_emb)[:, :, None, None]
        else:
            return self.conv_model(x)


class CouplingLayerConv(nn.Module):
    def __init__(self, c_in, kernel_size, cond_dim, mask_type, n_filter, n_hidden,
                 activation=nn.Tanh(), wn=True, num_res_blocks=3, c_extra=None):
        super().__init__()
        self.c_in = c_in
        self.c_extra = c_extra  # for VarDQ
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.cond_dim = cond_dim
        self.mask_type = mask_type
        self.activation = activation
        self.wn = wn
        self.num_res_blocks = num_res_blocks
        self.n_hidden = n_hidden
        self.register_buffer('tiny_mask', torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float))
        self.scaling_factor = nn.Parameter(torch.zeros(self.c_in))
        if c_extra is not None and cond_dim is not None:
            raise ValueError('Either conditional conv or VarDQ')

        # now define conv layers
        self.model_dict = nn.ModuleDict({'m1': MyConv(c_in=self.c_in if self.c_extra is None else self.c_in+self.c_extra,
                                                      cond_dim=self.cond_dim, activation=self.activation,
                                                      n_filter=self.n_filter, kernel_size=1,
                                                      n_hidden=self.n_hidden, layer=1, wn=self.wn)})
        for _ in range(self.num_res_blocks-1):
            self.model_dict['m2_{}'.format(_)] = MyConv(c_in=self.n_filter, cond_dim=self.cond_dim,
                                                        activation=self.activation, n_filter=self.n_filter,
                                                        kernel_size=self.kernel_size, n_hidden=self.n_hidden,
                                                        layer=1, wn=self.wn)
            self.model_dict['m3_{}'.format(_)] = MyConv(c_in=self.n_filter, cond_dim=self.cond_dim,
                                                        activation=None, n_filter=self.n_filter,
                                                        kernel_size=self.kernel_size, n_hidden=self.n_hidden,
                                                        layer=1, wn=self.wn)
        self.model_dict['shift'] = MyConv(c_in=self.n_filter, cond_dim=self.cond_dim, activation=None,
                                          n_filter=self.c_in, kernel_size=1, n_hidden=self.n_hidden,
                                          layer=1, wn=False)
        self.model_dict['scale'] = MyConv(c_in=self.n_filter, cond_dim=self.cond_dim, activation=self.activation,
                                          n_filter=self.c_in, kernel_size=1, n_hidden=self.n_hidden,
                                          layer=1, wn=False)

    def get_mask(self, x_shape, mask_type):
        # H, W of a fig have to be even
        assert self.mask_type in ['checkerboard0', 'checkerboard1', 'channel0', 'channel1']
        if 'checkerboard' in mask_type:
            unit0 = self.tiny_mask
            unit1 = 1.0 - unit0
            unit = unit0 if mask_type == 'checkerboard0' else unit1
            unit = torch.reshape(unit, [1, 1, 2, 2])
            if x_shape[2]%2 == 0 and x_shape[3]%2 == 0:
                b = torch.tile(unit, [x_shape[0], x_shape[1], x_shape[2] // 2, x_shape[3] // 2])
            else:
                b = torch.tile(unit, [x_shape[0], x_shape[1], x_shape[2] // 2 + 1, x_shape[3] // 2 + 1])
                b = b[:, :, :x_shape[2], :x_shape[3]]
        else:
            alter_seq = torch.arange(x_shape[1], device=self.tiny_mask.device) % 2
            if mask_type == 'channel0':
                b = (alter_seq[None, :, None, None] *
                     torch.ones([x_shape[0], x_shape[1], x_shape[2], x_shape[3]], device=self.tiny_mask.device))
            else:
                b = 1. - (alter_seq[None, :, None, None] *
                          torch.ones([x_shape[0], x_shape[1], x_shape[2], x_shape[3]], device=self.tiny_mask.device))
        return b

    def forward(self, x, y, task, log_det, z, x_addition=None):
        xs = list(x.shape)
        mask = self.get_mask(xs, self.mask_type)
        masked_x = x * mask

        # do a homemade resnet thing
        if x_addition is not None and self.cond_dim is None:
            cond_emb = None
            curr = self.model_dict['m1'](torch.cat((masked_x, x_addition), dim=1), cond_emb)
        elif x_addition is None and self.cond_dim is not None:
            cond_emb = y if task is None else torch.hstack((y, task))
            curr = self.model_dict['m1'](masked_x, cond_emb)
        else:
            raise ValueError('Either no cond_dim or no x_addition for VarDQ')
        skip = curr
        for _ in range(self.num_res_blocks-1):
            curr = self.model_dict['m2_{}'.format(_)](curr, cond_emb)
            curr = self.model_dict['m3_{}'.format(_)](curr, cond_emb)
            curr += skip
            curr = self.activation(curr)
            skip = curr
        shift = self.model_dict['shift'](curr, cond_emb) * (1-mask)
        # TODO: add a scaling factor
        scale = self.model_dict['scale'](curr, cond_emb) * (1-mask)
        s_fac = self.scaling_factor.exp()[None, :, None, None]
        scale = torch.tanh(scale / s_fac) * s_fac

        res = masked_x + (1. - mask) * (x * torch.exp(scale) + shift)
        log_det += scale.sum([1, 2, 3])

        return res, log_det, z

    def inverse(self, u, y, task, log_det, z, x_addition=None):
        us = list(u.shape)
        mask = self.get_mask(us, self.mask_type)
        masked_u = u * mask

        # do a homemade resnet thing
        if x_addition is not None and self.cond_dim is None:
            cond_emb = None
            curr = self.model_dict['m1'](torch.cat((masked_u, x_addition), dim=1), cond_emb)
        elif x_addition is None and self.cond_dim is not None:
            cond_emb = y if task is None else torch.hstack((y, task))
            curr = self.model_dict['m1'](masked_u, cond_emb)
        else:
            raise ValueError('Either no cond_dim or no x_addition for VarDQ')

        skip = curr
        for _ in range(self.num_res_blocks-1):
            curr = self.model_dict['m2_{}'.format(_)](curr, cond_emb)
            curr = self.model_dict['m3_{}'.format(_)](curr, cond_emb)
            curr += skip
            curr = self.activation(curr)
            skip = curr
        shift = self.model_dict['shift'](curr, cond_emb) * (1-mask)
        # scale = self.model_dict['scale'](curr, cond_emb) * (1-mask)
        # TODO: add a scaling factor
        scale = self.model_dict['scale'](curr, cond_emb) * (1-mask)
        s_fac = self.scaling_factor.exp()[None, :, None, None]
        scale = torch.tanh(scale / s_fac) * s_fac

        res = masked_u + (1. - mask) * (torch.exp(-1. * scale) * (u - shift))
        log_det += -1. * scale.sum([1, 2, 3])
        return res, log_det, z


class SqueezeLayer(nn.Module):
    def forward(self, x, y, task, log_det, z):
        B, C, H, W = x.shape
        # Forward direction: H x W x C => H/2 x W/2 x 4C
        x = x.reshape(B, C, H//2, 2, W//2, 2)
        x = x.permute(0, 1, 3, 5, 2, 4)
        x = x.reshape(B, 4*C, H//2, W//2)
        if z is not None and z.numel() != 0:
            z = z.reshape(B, C, H // 2, 2, W // 2, 2)
            z = z.permute(0, 1, 3, 5, 2, 4)
            z = z.reshape(B, 4 * C, H // 2, W // 2)
        return x, log_det, z

    def inverse(self, u, y, task, log_det, z):
        B, C, H, W = u.shape
        # Reverse direction: H/2 x W/2 x 4C => H x W x C
        u = u.reshape(B, C//4, 2, 2, H, W)
        u = u.permute(0, 1, 4, 2, 5, 3)
        u = u.reshape(B, C//4, H*2, W*2)
        if z is not None and z.numel() != 0:
            z = z.reshape(B, C//4, 2, 2, H, W)
            z = z.permute(0, 1, 4, 2, 5, 3)
            z = z.reshape(B, C//4, H*2, W*2)
        return u, log_det, z


class SplitLayer(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x, y, task, log_det, z):
        xs = list(x.shape)
        split = xs[1] // 2
        new_z = x[:, :split, :, :]
        x = x[:, split:, :, :]
        if z is not None:
            z = torch.concat([z, new_z], 1)
        else:
            z = new_z
        return x, log_det, z

    def inverse(self, u, y, task, log_det, z):
        zs = list(z.shape)
        if u is None:
            split = zs[1] // (2 ** self.scale)
        else:
            split = u.shape[1]
        new_u = z[:, -split:, :, :]
        z = z[:, :-split, :, :]
        assert (new_u.shape[1] == split)
        if u is not None:
            x = torch.concat([new_u, u], 1)
        else:
            x = new_u
        return x, log_det, z

#
# class VarDQ(nn.Module):
#     def __init__(self, H, W, C_in, alpha=1e-5, quants=256., y_dim=None, task_dim=None, cond_dim=None,
#                  n_res_block=3, n_filter=64, kernel_size=3, n_hidden=128, activation=nn.Tanh()):
#         super().__init__()
#         self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float))
#         self.register_buffer('quants', torch.tensor(quants, dtype=torch.float))
#         self.H, self.W, self.C_in = H, W, C_in
#         self.y_dim = y_dim
#         self.kernel_size = kernel_size
#         self.task_dim = task_dim
#         self.n_hidden = n_hidden
#         self.n_res_block = n_res_block
#         self.n_filter = n_filter
#         self.cond_dim = cond_dim
#         self.activation = activation
#         self.c_extra = C_in
#         my_flow = []
#         my_flow.append(CouplingLayerConv(c_in=self.C_in, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
#                                          mask_type='checkerboard0', n_filter=self.n_filter, n_hidden=self.n_hidden,
#                                          activation=self.activation, wn=True, num_res_blocks=self.n_res_block,
#                                          c_extra=self.c_extra))
#         my_flow.append(CouplingLayerConv(c_in=self.C_in, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
#                                          mask_type='checkerboard1', n_filter=self.n_filter, n_hidden=self.n_hidden,
#                                          activation=self.activation, wn=True, num_res_blocks=self.n_res_block,
#                                          c_extra=self.c_extra))
#         my_flow.append(CouplingLayerConv(c_in=self.C_in, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
#                                          mask_type='checkerboard0', n_filter=self.n_filter, n_hidden=self.n_hidden,
#                                          activation=self.activation, wn=True, num_res_blocks=self.n_res_block,
#                                          c_extra=self.c_extra))
#         my_flow.append(CouplingLayerConv(c_in=self.C_in, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
#                                          mask_type='checkerboard1', n_filter=self.n_filter, n_hidden=self.n_hidden,
#                                          activation=self.activation, wn=True, num_res_blocks=self.n_res_block,
#                                          c_extra=self.c_extra))
#         self.my_flow = nn.ModuleList(my_flow)
#
#     def forward(self, x, log_det, z):  # dequant to -infty, infty
#         x, log_det, z = self.dq(x, log_det, z)
#         x, log_det, z = self.logit(x, log_det, z)
#         return x, log_det, z
#     # TODO: first without VarDQ, use standard dq
#     # ToDO: we want the dequantized logit to match the rvp inv?
#     # pass latent to generate logit values using the current model
#     # then pass the generated img to curr varDQ to get logit values under current model
#     # then compare the same thing under the reference model
#     def inverse(self, u, log_det, z):
#         u, log_det, z = self.sigmoid(u, log_det, z)
#         u = u * self.quants
#         log_det = log_det + torch.tensor(np.log(self.quants.cpu().item()) * np.prod(u.shape[1:]),
#                                          device=self.alpha.device, dtype=torch.float)
#         u = torch.floor(u).clamp(min=0, max=self.quants - 1).to(torch.int32)
#         return u, log_det, z
#
#     def sigmoid(self, x, log_det, z):
#         log_det = log_det + (-x - 2 * torch.nn.functional.softplus(-x)).sum(dim=[1, 2, 3])
#         x = torch.sigmoid(x)
#         # Reversing scaling for numerical stability
#         log_det = log_det - torch.log(1 - self.alpha) * torch.tensor(np.prod(x.shape[1:]),
#                                                                      dtype=torch.float, device=self.alpha.device)
#         x = (x - 0.5 * self.alpha) / (1 - self.alpha)
#         return x, log_det, z
#
#     def logit(self, u, log_det, z):
#         u = u * (1 - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
#         log_det = log_det + torch.log(1 - self.alpha) * torch.tensor(np.prod(u.shape[1:]),
#                                                                      dtype=torch.float, device=self.alpha.device)
#         log_det = log_det + (-torch.log(u) - torch.log(1 - u)).sum(dim=[1, 2, 3])
#         u = torch.log(u) - torch.log(1 - u)
#         return u, log_det, z
#
#     def dq(self, x, log_det, z):
#         x = x.to(torch.float)
#         img = (x / (self.quants-1.)) * 2 - 1  # We condition the flows on the original image
#         # Prior of u is a uniform distribution as before
#         # As most flow transformations are defined on [-infinity,+infinity], we apply an inverse sigmoid first.
#         deq_noise = torch.rand_like(x).detach()
#         # deq_noise = torch.arange(np.prod(x.shape)).reshape(x.shape)/np.prod(x.shape)
#         deq_noise, log_det, z = self.logit(deq_noise, log_det, z)
#         for flow in self.my_flow:
#             deq_noise, log_det, z = flow(deq_noise, None, None, log_det, z, x_addition=img)
#         deq_noise, log_det, z = self.sigmoid(deq_noise, log_det, z)
#         # After the flows, apply u as in standard dq
#         x = (x + deq_noise) / self.quants
#         log_det = log_det - torch.tensor(np.log(256.0) * np.prod(x.shape[1:]),
#                                          device=self.alpha.device, dtype=torch.float)
#         return x, log_det, z


class CRealNVPConv(nn.Module):
    def __init__(self, H, W, C_in, y_dim=None, task_dim=None, cond_dim=10, n_res_block=3, n_filter=64, kernel_size=3,
                 n_hidden=128, activation=nn.Tanh(), scale=3, var_dq=None):
        # base density should be a nn.module, gotta send it to device manually tho
        super(CRealNVPConv, self).__init__()
        self.activation = activation
        self.H = H
        self.W = W
        self.C_in = C_in
        self.H_out, self.W_out, self.C_out = H, W, C_in
        self.y_dim = y_dim
        self.kernel_size = kernel_size
        self.task_dim = task_dim
        self.n_hidden = n_hidden
        self.n_res_block = n_res_block
        self.n_filter = n_filter
        self.cond_dim = self.task_dim + self.y_dim
        self.scale = scale
        self.var_dq = var_dq
        self.varDQ = None
        if self.var_dq is not None:
            raise ValueError('Do not use VarDQ first!')
            # self.varDQ = VarDQ(H, W, C_in, alpha=1e-5, quants=256., y_dim=None, task_dim=None, cond_dim=None,
            #                    n_res_block=2, n_filter=64, kernel_size=3, n_hidden=128, activation=nn.Tanh())

        curr_channel = self.C_in
        net = []
        for curr_scale in range(self.scale-1):
            net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                         mask_type='checkerboard0', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                         activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
            net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                         mask_type='checkerboard1', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                         activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
            net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                         mask_type='checkerboard0', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                         activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
            net.append(SqueezeLayer())  # H, W halved, channel * 4
            curr_channel = curr_channel * 4
            self.H_out, self.W_out, self.C_out = self.H_out//2, self.W_out//2, self.C_out*4
            net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                         mask_type='channel0', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                         activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
            net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                         mask_type='channel1', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                         activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
            net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                         mask_type='channel0', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                         activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
            net.append(SplitLayer(curr_scale))  # channel halved
            curr_channel = curr_channel // 2
        net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                     mask_type='checkerboard0', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                     activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
        net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                     mask_type='checkerboard1', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                     activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
        net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                     mask_type='checkerboard0', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                     activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
        net.append(CouplingLayerConv(c_in=curr_channel, kernel_size=self.kernel_size, cond_dim=self.cond_dim,
                                     mask_type='checkerboard1', n_filter=self.n_filter, n_hidden=self.n_hidden,
                                     activation=self.activation, wn=True, num_res_blocks=self.n_res_block))
        net.append(SplitLayer(scale-1))  # channel halved

        self.net = nn.ModuleList(net)

    def forward(self, x, y, task=None):  # maps obs x to base u given y and task
        log_det = 0.
        z = None
        if self.var_dq is not None:
            # turn 0, 256 to -infty. infty
            x, log_det, z = self.varDQ(x, log_det, z)
        for coupling in self.net:
            x, log_det, z = coupling(x, y, task, log_det, z)
        return torch.concat((z, x), 1), log_det
        # x batch size*(C*H*W) x --> size*(new CHW), log_det has length batch size

    def inverse(self, u, y, task=None):  # maps base u to target x
        log_det = 0.
        z = None
        for coupling in reversed(self.net):
            z, log_det, u = coupling.inverse(z, y, task, log_det, u)
        if self.var_dq is not None:
            # turn -infty, infty to 0, 256
            z, log_det, u = self.varDQ.inverse(z, log_det, u)
        return z, log_det   # x batch size*(C*H*W) x --> size*(new CHW), log_det has length batch size


class MyDense(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.Tanh(), n_hidden=128, layer=3, wn=False):
        super().__init__()
        if wn:
            model = [weight_norm(nn.Linear(dim_in, n_hidden), name='weight')]
            for _ in range(layer - 1):
                model += [activation, weight_norm(nn.Linear(n_hidden, n_hidden), name='weight')]
            model += [activation, weight_norm(nn.Linear(n_hidden, dim_out), name='weight')]
            self.model = nn.Sequential(*model)
        else:
            model = [nn.Linear(dim_in, n_hidden)]
            for _ in range(layer-1):
                model += [activation, nn.Linear(n_hidden, n_hidden)]
            model += [activation, nn.Linear(n_hidden, dim_out)]
            self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class CouplingLayerNVP(nn.Module):
    def __init__(self, mask, activation=nn.Tanh(), x_dim=None, y_dim=None, task_dim=None, cond_dim=None, n_hidden=128):
        super().__init__()
        self.register_buffer('mask', mask)
        self.activation = activation
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.task_dim = task_dim
        self.n_hidden = n_hidden
        # todo: trainable or fixed scaling factor?
        self.scaling_factor = nn.Parameter(torch.zeros(1))
        # self.register_buffer('scaling_factor', torch.tensor(0.1, dtype=torch.float))

        # define the conditioner encoder
        if cond_dim is None:
            self.cond_dim = x_dim
        else:
            self.cond_dim = cond_dim
        self.cond_net = None
        if task_dim is None:
            self.cond_net = MyDense(self.y_dim, self.cond_dim, self.activation, self.n_hidden, wn=True)
        elif y_dim is not None and task_dim is not None:
            self.cond_net = MyDense(self.y_dim + self.task_dim, self.cond_dim, self.activation, self.n_hidden, wn=True)
        else:
            raise ValueError("either task_dim=None for single task or both y_dim and task_dim are not None")

        # real NVP transformation, input should be cat(mask*x, task_embedding, category embedding)
        # self.scale_net = MyDense(self.x_dim + self.cond_dim, self.x_dim, self.activation, self.n_hidden)
        # self.shift_net = MyDense(self.x_dim + self.cond_dim, self.x_dim, self.activation, self.n_hidden)
        self.net = MyDense(self.x_dim + self.cond_dim, 2*self.x_dim, self.activation, self.n_hidden)

    def inverse(self, u, y, task=None):  # data to noise
        if task is None and self.cond_dim != u.shape[-1]//2:
            raise ValueError("inconsistent specification of task")
        conditioner = self.cond_net(y if task is None else torch.hstack((y, task)))
        masked_u = u * self.mask
        aug_u = torch.hstack((masked_u, conditioner))
        # TODO: add a scaling factor
        # shift = self.shift_net(aug_u) * (1. - self.mask)
        # scale = self.scale_net(aug_u) * (1. - self.mask)
        ans = self.net(aug_u)
        shift = ans[:, :self.x_dim] * (1. - self.mask)
        scale = ans[:, self.x_dim:] * (1. - self.mask)

        s_fac = self.scaling_factor.exp()
        scale = torch.tanh(scale / s_fac) * s_fac
        x = masked_u + (1 - self.mask) * (torch.exp(-1. * scale) * (u - shift))
        log_det_jac = -1. * scale

        return x, log_det_jac

    def forward(self, x, y, task=None):
        if task is None and self.cond_dim != x.shape[-1] // 2:
            raise ValueError("inconsistent specification of task")
        conditioner = self.cond_net(y if task is None else torch.hstack((y, task)))
        masked_x = x * self.mask
        aug_x = torch.hstack((masked_x, conditioner))
        # TODO: add a scaling factor
        # shift = self.shift_net(aug_x) * (1. - self.mask)
        # scale = self.scale_net(aug_x) * (1. - self.mask)
        ans = self.net(aug_x)
        shift = ans[:, :self.x_dim] * (1. - self.mask)
        scale = ans[:, self.x_dim:] * (1. - self.mask)

        s_fac = self.scaling_factor.exp()
        scale = torch.tanh(scale / s_fac) * s_fac

        u = masked_x + (1 - self.mask) * (x * torch.exp(scale) + shift)
        log_det_jac = scale

        return u, log_det_jac


class CRealNVP(nn.Module):
    def __init__(self, H=None, W=None, C_in=None, x_dim=None, y_dim=None, task_dim=None, cond_dim=10, conv=False,
                 n_res_block=6, n_dense_block=6, n_filter=64, kernel_size=3, n_hidden_conv=128,
                 n_hidden_dense=128, activation=nn.Tanh(), scale=3, var_dq=None):
        # base density should be a nn.module, gotta send it to device manually tho
        super(CRealNVP, self).__init__()
        self.activation = activation
        self.H = H
        self.W = W
        self.C_in = C_in
        self.H_out, self.W_out, self.C_out = None, None, None
        if x_dim is None:
            self.x_dim = H*W*C_in
        else:
            self.x_dim = x_dim
        self.y_dim = y_dim
        self.task_dim = task_dim
        self.n_hidden = n_hidden_conv
        self.n_hidden_dense = n_hidden_dense
        self.n_dense_block = n_dense_block
        self.register_buffer('mask', torch.arange(self.x_dim, dtype=torch.float) % 2)
        self.cond_dim = cond_dim
        self.n_res_block = n_res_block
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.scale = scale
        self.var_dq = var_dq

        net_dense = [CouplingLayerNVP(mask=self.mask, activation=self.activation, x_dim=self.x_dim, y_dim=self.y_dim,
                                      task_dim=self.task_dim, cond_dim=self.cond_dim, n_hidden=self.n_hidden_dense)]
        for _ in range(self.n_dense_block - 1):
            self.mask = 1 - self.mask
            net_dense += [CouplingLayerNVP(mask=self.mask, activation=self.activation, x_dim=self.x_dim,
                                           y_dim=self.y_dim, task_dim=self.task_dim, cond_dim=self.cond_dim,
                                           n_hidden=self.n_hidden_dense)]
        self.net_dense = nn.ModuleList(net_dense)

        if conv:
            self.net_conv = nn.ModuleList([CRealNVPConv(H=self.H, W=self.W, C_in=self.C_in,
                                                        y_dim=self.y_dim, task_dim=self.task_dim,
                                                        cond_dim=self.cond_dim, n_res_block=self.n_res_block,
                                                        n_filter=self.n_filter, kernel_size=self.kernel_size,
                                                        n_hidden=self.n_hidden, activation=self.activation,
                                                        scale=self.scale, var_dq=self.var_dq)])
            self.H_out, self.W_out, self.C_out = self.net_conv[0].H_out, self.net_conv[0].W_out, self.net_conv[0].C_out
        else:
            self.net_conv = None

    def forward(self, x, y, task=None):  # maps obs x to base u given y and task
        if self.net_conv is not None:
            # x, log_det_dq = dequantization_forward_and_jacobian(x)
            # x, log_det_logit = logit_transformation(x)
            x, log_det = self.net_conv[0](x, y, task)
            x = x.reshape(x.shape[0], self.H_out*self.W_out*self.C_out)
            # log_det = log_det + log_det_dq + log_det_logit
        else:
            log_det = 0.
        for coupling in self.net_dense:
            x, det = coupling(x, y, task)
            log_det += det.sum(1)
        return x, log_det  # both x and log det has shape batch size*(old_par_size) x-->u

    def inverse(self, u, y, task=None):  # maps base u to target x
        log_det = 0.
        for coupling in reversed(self.net_dense):
            u, det = coupling.inverse(u, y, task)
            log_det += det.sum(1)
        if self.net_conv is None:
            return u, log_det
        else:
            u = u.reshape(u.shape[0], self.C_out, self.H_out, self.W_out)
            res, conv_log_det = self.net_conv[0].inverse(u, y, task)
            return res, log_det + conv_log_det
            # return torch.floor(torch.nn.functional.sigmoid(res) * 256.), log_det + conv_log_det


def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def inv_sigmoid(x):
    return torch.log(x) - torch.log(1. - x)


class GaussianRecurrent(nn.Module):
    def __init__(self, x_dim, mu_init=0., var_init=1., corr_init=0.1):
        super().__init__()
        self.dim = x_dim

        self.register_buffer('prior_mu', mu_init * torch.ones(x_dim, dtype=torch.float))
        self.register_buffer('pi', torch.tensor(torch.pi, dtype=torch.float))
        self.var_vbl = nn.Parameter(inv_softplus(torch.sqrt(var_init * torch.ones(x_dim))))
        self.corr_vbl = nn.Parameter(inv_sigmoid(corr_init * torch.ones(x_dim)))

        self.curr_var = torch.square(nn.Softplus()(self.var_vbl))
        self.curr_mu = self.prior_mu * 1.0

        self.sample_size = 0
        self.z_sum = torch.zeros(self.dim, dtype=torch.float, device=self.prior_mu.device)

        # check point for restart, default = prior, can be updated
        self.var_restart = torch.square(nn.Softplus()(self.var_vbl))
        self.mu_restart = self.prior_mu * 1.0
        self.sample_size_restart = 0
        self.z_sum_restart = torch.zeros(self.dim, dtype=torch.float, device=self.prior_mu.device)

    def get_prior_var(self):
        return torch.square(nn.Softplus()(self.var_vbl))

    def get_prior_cov(self):
        return torch.sigmoid(self.corr_vbl) * self.get_prior_var()

    def update_distribution(self, x):  # update given a single obs
        self.z_sum = self.z_sum + x - self.prior_mu
        self.sample_size += 1

        prior_cov = self.get_prior_cov()
        prior_var = self.get_prior_var()

        dd = prior_cov / (prior_var + prior_cov * (self.sample_size - 1.))
        self.curr_mu = (1. - dd) * self.curr_mu + x * dd
        self.curr_var = (1. - dd) * self.curr_var + (prior_var - prior_cov) * dd

    def reset(self):
        self.curr_var = self.var_restart * 1.0
        self.curr_mu = self.mu_restart * 1.0
        self.sample_size = self.sample_size_restart * 1
        self.z_sum = self.z_sum_restart * 1.0

    def update_restart(self):
        self.var_restart = self.curr_var * 1.0
        self.mu_restart = self.curr_mu * 1.0
        self.sample_size_restart = self.sample_size * 1
        self.z_sum_restart = self.z_sum * 1.0

    def get_log_likelihood(self, x, mask_dim=None):  # given a single obs
        log_pdf = -0.5 * torch.log(2. * self.pi * self.curr_var) - torch.square(x - self.curr_mu) / (2. * self.curr_var)
        if mask_dim is not None:
            return (log_pdf * mask_dim).sum()
        else:
            return log_pdf.sum()

    def get_log_likelihood_under_prior(self, x, mask_dim=None):  # given a single obs
        mu, var = self.prior_mu, self.get_prior_var()
        log_pdf = -0.5 * torch.log(2. * self.pi * var) - torch.square(x - mu) / (2. * var)
        if mask_dim is not None:
            return (log_pdf * mask_dim).sum()
        else:
            return log_pdf.sum()

    def init_train_lkd(self, z_rest, z_context=None):
        curr_mu = self.prior_mu * 1.0
        curr_var = self.get_prior_var() * 1.0
        prior_var = self.get_prior_var() * 1.0
        prior_cov = self.get_prior_cov() * 1.0
        sample_size = self.sample_size*1.0
        lkd = 0.
        if z_context is None:
            for i in range(z_rest.shape[0]):
                lkd += (-0.5 * torch.log(2. * self.pi * curr_var) -
                        torch.square(z_rest[i] - curr_mu) / (2. * curr_var)).sum()
                sample_size += 1
                dd = prior_cov / (prior_var + prior_cov * (sample_size - 1.))
                curr_mu = (1. - dd) * curr_mu + z_rest[i] * dd
                curr_var = (1. - dd) * curr_var + (prior_var - prior_cov) * dd
            return lkd
        else:
            for i in range(z_context.shape[0]):
                sample_size += 1
                dd = prior_cov / (prior_var + prior_cov * (sample_size - 1.))
                curr_mu = (1. - dd) * curr_mu + z_context[i] * dd
                curr_var = (1. - dd) * curr_var + (prior_var - prior_cov) * dd
            for i in range(z_rest.shape[0]):
                lkd += (-0.5 * torch.log(2. * self.pi * curr_var) -
                        torch.square(z_rest[i] - curr_mu) / (2. * curr_var)).sum()
            return lkd

    def ongoing_prior(self, z_rest, z_context=None):
        curr_mu = self.curr_mu * 1.0
        curr_var = self.curr_var * 1.0
        prior_var = self.get_prior_var() * 1.0
        prior_cov = self.get_prior_cov() * 1.0
        sample_size = self.sample_size*1.0
        lkd = 0.
        if z_context is None:
            for i in range(z_rest.shape[0]):
                lkd += (-0.5 * torch.log(2. * self.pi * curr_var) -
                        torch.square(z_rest[i] - curr_mu) / (2. * curr_var)).sum()
                sample_size += 1
                dd = prior_cov / (prior_var + prior_cov * (sample_size - 1.))
                curr_mu = (1. - dd) * curr_mu + z_rest[i] * dd
                curr_var = (1. - dd) * curr_var + (prior_var - prior_cov) * dd
            return lkd
        else:
            for i in range(z_context.shape[0]):
                sample_size += 1
                dd = prior_cov / (prior_var + prior_cov * (sample_size - 1.))
                curr_mu = (1. - dd) * curr_mu + z_context[i] * dd
                curr_var = (1. - dd) * curr_var + (prior_var - prior_cov) * dd
            for i in range(z_rest.shape[0]):
                lkd += (-0.5 * torch.log(2. * self.pi * curr_var) -
                        torch.square(z_rest[i] - curr_mu) / (2. * curr_var)).sum()
            return lkd

    def sample(self, n, prior=False):  # sample given current mu and var
        if prior:
            return torch.randn((n, self.dim)) * torch.sqrt(self.get_prior_var()[None, :]) + self.prior_mu[None, :]
        else:
            return torch.randn((n, self.dim)) * torch.sqrt(self.curr_var[None, :]) + self.curr_mu[None, :]


class MyDataset(Dataset):
    # merge all datasets, use task to identify different tasks
    def __init__(self, X, y, task):
        self.X = X
        self.y = y
        self.task = task

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # vec, int, int
        return self.X[idx], self.y[idx], self.task[idx]


class CLBruno(nn.Module):
    def __init__(self, H=None, W=None, C_in=None, x_dim = None, y_dim=None, task_dim=None,
                 cond_dim=None, conv=False, task_num=None, y_cat_num=None, single_task=False, var_dq=None,
                 n_res_block=6, scale=3, n_dense_block=6, n_filter=64, kernel_size=3,
                 n_hidden_conv=128, n_hidden_dense=128, activation=nn.Tanh(), mu_init=0., var_init=1., corr_init=0.1):
        # need sampling_mode to generate pseudo samples, encode each task and category as a (N01)^y_dim, task_dim vec,
        # add regularization to these embeddings, these are additional lists and dicts

        # x_dim=dim of x. y_dim = dim of cat embedding, task_dim=dim of task embedding
        # task num = # of initial tasks, y_cat_num=list of # of cats for each task
        super().__init__()
        self.cond_dim = cond_dim
        if x_dim is None:
            self.x_dim = H * W * C_in
        else:
            self.x_dim = x_dim
        self.conv = conv
        self.n_res_block = n_res_block
        self.scale = scale
        self.n_dense_block = n_dense_block
        self.var_dq = var_dq
        self.n_filter = n_filter
        self.kernel_size = kernel_size
        self.n_hidden_conv = n_hidden_conv
        self.n_hidden_dense = n_hidden_dense
        self.y_emb_dim = y_dim
        self.task_emb_dim = task_dim

        self.c_nvp = CRealNVP(H=H, W=W, C_in=C_in, x_dim=self.x_dim, y_dim=self.y_emb_dim, task_dim=self.task_emb_dim,
                              cond_dim=self.cond_dim, conv=self.conv, n_res_block=self.n_res_block,
                              n_dense_block=self.n_dense_block, n_filter=self.n_filter, kernel_size=self.kernel_size,
                              n_hidden_conv=self.n_hidden_conv, n_hidden_dense=self.n_hidden_dense,
                              activation=activation, scale=self.scale, var_dq=self.var_dq)
        self.base_list = nn.ModuleList([GaussianRecurrent(x_dim=self.x_dim, mu_init=mu_init, var_init=var_init,
                                                          corr_init=corr_init) for _ in range(task_num)])
        # prior layers should be fixed after initial training
        self.task_id_list = list(range(task_num))
        self.y_cat_num = y_cat_num  # should be a list of cat nums associated with each task
        self.activation = activation
        self.mu_init = mu_init
        self.corr_init = corr_init
        self.var_init = var_init

        # initialize task and category embedding
        if not single_task:
            self.task_emb = nn.Parameter(torch.randn(task_num, task_dim))
            self.y_emb = nn.ParameterList(
                [nn.Parameter(torch.randn(y_cat_num[_], y_dim)) for _ in range(task_num)])
        else:
            if task_num != 1:
                raise ValueError('`only one task allowed')
            # mute task embedding if only one task
            self.task_emb = nn.Parameter(torch.zeros(1, task_dim))
            self.y_emb = nn.ParameterList(
                [nn.Parameter(torch.randn(y_cat_num[_], y_dim)) for _ in range(task_num)])

        # store reference model for continual training
        self.reference_base = None
        self.reference_model = None
        self.reference_task_emb = None
        self.reference_y_emb = None

        # record prior label prob
        self.prior_label = nn.ParameterList([torch.ones(y_cat_num[_])/(y_cat_num[_]*1.0) for _ in range(task_num)])
        for _ in self.prior_label:
            _.requires_grad = False

    def task_specific_lkd(self, x, y, task=None, n_context_portion=0.2, reference=False):  # handles single task, task=int task id
        # NEEDS TO RESET PRIOR LAYERS BEFORE AND AFTER TRAINING!
        # y is vector integer categorical under task t
        # t is a int task id
        if reference:
            task_emb = self.reference_task_emb[task].unsqueeze(0).repeat(x.shape[0], 1)
            y_emb = self.reference_y_emb[task][y]
            z, log_det_jac = self.reference_model(x, y_emb, task_emb)
        else:
            task_emb = self.task_emb[task].unsqueeze(0).repeat(x.shape[0], 1)
            y_emb = self.y_emb[task][y]
            z, log_det_jac = self.c_nvp(x, y_emb, task_emb)
        log_prior_prob = 0.
        task_base = self.base_list[task]
        # task_base.reset()  # reset the prior layer to check point before loss eval, for initial training is prior
        # for continual regularization is the updated prior
        if n_context_portion > 0:
            # if we look at predictive of (n_seq - n_context)|n_context
            n_context = int(n_context_portion * x.shape[0])
            log_prior_prob += task_base.ongoing_prior(z_rest=z[n_context:], z_context=z[:n_context])
            # n_seq = x.shape[0]
            # for i in range(n_context):
            #     task_base.update_distribution(z[i])
            # for i in range(n_context, n_seq):
            #     log_prior_prob += task_base.get_log_likelihood(z[i])
            # task_base.reset()  # reset the prior layer after loss eval
            return log_det_jac[n_context:].sum() + log_prior_prob
            # averaged over all dimensions and non-context test samples (n_seq-n_context of them)
        else:
            log_prior_prob += task_base.ongoing_prior(z_rest=z, z_context=None)
            # n_seq = x.shape[0]
            # for i in range(n_seq):
            #     log_prior_prob += task_base.get_log_likelihood(z[i])
            #     task_base.update_distribution(z[i])
            # task_base.reset()  # reset the prior layer after loss eval
            return log_det_jac.sum() + log_prior_prob

    def initial_prior_lkd(self, x, y, task=None, n_context_portion=0.2):  # handles single task, task=int task id
        # used to train a new task with a new prior layer
        # NEEDS TO RESET PRIOR LAYERS BEFORE AND AFTER TRAINING!
        # y is vector integer categorical under task t
        # t is a int task id
        task_emb = self.task_emb[task].unsqueeze(0).repeat(x.shape[0], 1)
        y_emb = self.y_emb[task][y]
        z, log_det_jac = self.c_nvp(x, y_emb, task_emb)
        log_prior_prob = 0.
        task_base = self.base_list[task]
        # task_base.reset()  # reset the prior layer to check point before loss eval, for initial training is prior
        # for continual regularization is the updated prior
        if n_context_portion > 0:
            # if we look at predictive of (n_seq - n_context)|n_context
            n_context = int(n_context_portion * x.shape[0])
            log_prior_prob += task_base.init_train_lkd(z_rest=z[n_context:], z_context=z[:n_context])
            # task_base.reset()  # reset the prior layer after loss eval
            return log_det_jac[n_context:].sum() + log_prior_prob
            # averaged over all dimensions and non-context test samples (n_seq-n_context of them)
        else:
            log_prior_prob += task_base.init_train_lkd(z_rest=z, z_context=None)
            # task_base.reset()  # reset the prior layer after loss eval
            return log_det_jac.sum() + log_prior_prob
        # todo: turn loss to per dim per sample, curr=all dim per sample
        # scaling of regularization is probably wrong, try think from a Bayesian prior-lkd perspective?

    def task_specific_sampling(self, task_id, n_sample, prior=False):
        # assuming prior layer has been updated, for generating pseudo-samples for task t
        p = self.prior_label[task_id].detach().cpu().numpy()
        y_cats = list(np.random.choice(range(len(p)), size=n_sample, replace=True, p=p))
        # sample pseudo samples from the reference, first draw samples from prior label,
        # then work out generated samples
        with torch.no_grad():
            z_samples = self.reference_base[task_id].sample(n_sample, prior=prior)
            y_samples = self.reference_y_emb[task_id][y_cats]
            task = self.reference_task_emb[[task_id] * n_sample]
            x, log_det_jac = self.reference_model.inverse(z_samples, y_samples, task)

        return z_samples, y_cats, x, log_det_jac

    def continual_regularization(self, n_pseudo, alignment_reg=1., norm_reg=1., prior=False):
        # regularization strength,
        # alignment_reg penalize deviation between nvp_old and nvp_curr over a fixed set of latent samples
        # prev_lkd encourages nvp_curr to induce the same distribution as nvp_old on previously seen cats and tasks
        # norm_reg penalizes deviation from curr vs reference task_ and cat_embedding
        alignment_reg = torch.tensor(alignment_reg, device=self.task_emb.device, dtype=torch.float)
        norm_reg = torch.tensor(norm_reg, device=self.task_emb.device, dtype=torch.float)
        # generating pseudo samples
        pseudo_noise = []
        pseudo_samples = []
        pseudo_cats = []
        with torch.no_grad():
            for task_id in self.task_id_list:
                z_samples, y_cats,  x, _ = self.task_specific_sampling(task_id=task_id, n_sample=n_pseudo, prior=prior)
                pseudo_noise.append(z_samples)
                pseudo_samples.append(x)
                pseudo_cats.append(y_cats)

        prev_lkd = 0.
        alignment = 0.
        norm = 0.
        for task_id in self.task_id_list:
            # log lkd regularization on pseudo samples
            # evaluate p_{curr_model}(old_samples), minimizing kl div between curr and reference model
            prev_lkd -= self.task_specific_lkd(x=pseudo_samples[task_id], y=pseudo_cats[task_id],
                                               task=task_id, n_context_portion=0.2, reference=False)/pseudo_samples[task_id].numel()

            # alignment regularization on generated samples
            # work out ||f_{curr_model}(z) - f_{reference_model}(z)||2
            x, curr_det = self.c_nvp.inverse(u=pseudo_noise[task_id], y=self.y_emb[task_id][pseudo_cats[task_id]],
                                             task=self.task_emb[[task_id]*n_pseudo])
            # generate pseudo samples using the same task, y_label and noises and the current model,
            # compare outcome with the reference model
            alignment += ((x - pseudo_samples[task_id])**2).mean()
            # todo: consider += _.mean(), equivalent to reverse KL as base models are the same?
            prev_lkd -= 0.1*self.task_specific_lkd(x=x, y=pseudo_cats[task_id],
                                               task=task_id, n_context_portion=0.2, reference=True)/x.numel()
            # norm += ((self.task_emb[task_id])**2).mean() + (self.y_emb[task_id]**2).mean(1).sum()

            # penalize changes in previously seen class and task embeddings
            # task-incremental learning: class # remains the same vs class-incremental learning: more classes
            reference_classes = self.reference_y_emb[task_id].shape[0]
            norm += (((self.task_emb[task_id] - self.reference_task_emb[task_id]) ** 2).mean() +
                     ((self.y_emb[task_id][:reference_classes] - self.reference_y_emb[task_id]) ** 2).mean(1).sum())
        return prev_lkd + alignment_reg * alignment + norm_reg * norm

    def train_init(self, X_init, y_init, task_init, lr=1e-3, epoch=200, batch_size=256, embedding_reg=0.1, weight_decay=0.):
        # X_init, y_init, task_init: combined initial data,
        # task, y are encoded by 0,1,2,...
        train_set, test_set = random_split(MyDataset(X_init, y_init, task_init), [0.8, 0.2])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss, test_loss = np.zeros(epoch), np.zeros(epoch)
        self.turn_on_prior_train()
        for ep in range(epoch):
            print(ep)
            tl = 0.
            for x, y, task in train_loader:
                loss = torch.tensor(0., dtype=torch.float, device=self.task_emb.device)
                optimizer.zero_grad()
                x = x.to(self.task_emb.device)
                for task_id in torch.unique(task):
                    # reset prior layer before training
                    id_vec = task == task_id
                    loss -= self.initial_prior_lkd(x=x[id_vec], y=y[id_vec],
                                                   task=task_id, n_context_portion=0.2)/x.numel()
                    loss += (embedding_reg *
                             (((self.task_emb[task_id])**2).mean() + (self.y_emb[task_id]**2).mean(1).sum()))
                loss.backward()
                optimizer.step()
                tl += loss.detach().cpu().numpy()
                # reset prior layer after grad eval
                for task_id in torch.unique(task):
                    self.base_list[task_id].reset()

            train_loss[ep] = tl/(len(train_set)/len(test_set))

            # work out test loss
            with torch.no_grad():
                tl = 0.
                for x, y, task in test_loader:
                    loss = torch.tensor(0., dtype=torch.float, device=self.task_emb.device)
                    x = x.to(self.task_emb.device)
                    for task_id in torch.unique(task):
                        # reset prior before training
                        self.base_list[task_id].reset()
                        id_vec = task == task_id
                        loss -= self.initial_prior_lkd(x=x[id_vec], y=y[id_vec],
                                                       task=task_id, n_context_portion=0.2)/x.numel()
                        tl += loss.detach().cpu().numpy()
                        # reset prior after training
                        self.base_list[task_id].reset()
                test_loss[ep] = tl
        # training done, now update prior layers and prior label probs for each task

        # updating prior labels
        for _ in self.task_id_list:
            task_label = y_init[task_init == _]  # task_label = [0,1,2,3,2,1,2,3,4,3,...], start from 0
            self.prior_label[_] = torch.tensor([torch.mean(1.0*(task_label == __)) for
                                                __ in range(len(torch.unique(task_label)))],
                                               device=self.task_emb.device, requires_grad=False)
        self.turn_off_prior_train()  # detach trained prior layer from graph, treat them as fixed
        for task_id in self.task_id_list:
            self.continual_update_task(new_x=X_init[task_init == task_id],
                                       new_y=y_init[task_init == task_id],
                                       task_id=task_id)
        # update the four components of the model: task emb, class emb, nvp and prior layers
        self.update_reference()

        return train_loss, test_loss

    def train_new_task(self, new_X, new_y, new_task, lr=1e-3, epoch=200, batch_size=256,
                       n_pseudo=256, lam=1., embedding_reg=0.1, weight_decay=0.):
        # when a new task comes in
        # initialize new embedding
        # if already seen 5 tasks, unique(new_task) should be 1,2,3,4,5,6,7,8,9,...
        if len(set(torch.unique(new_task)) & set(self.task_id_list)) != 0:
            raise ValueError('only take disjoint tasks!data from already seen task should go to continual_update_task')
        new_task_num = len(set(torch.unique(new_task)) - set(self.task_id_list))
        new_task_id = list(range(len(self.task_id_list), len(self.task_id_list)+new_task_num))
        self.task_emb = nn.Parameter(torch.concat((self.reference_task_emb * 1.0,
                                                   torch.randn(new_task_num, self.reference_task_emb.shape[1]))))
        for new_id in new_task_id:
            self.y_cat_num.append(len(torch.unique(new_y[new_task == new_id])))
            self.y_emb.append(nn.Parameter(torch.randn(self.y_cat_num[new_id], self.y_emb_dim)))
        # initialize new base
        for _ in range(new_task_num):
            self.base_list.append(GaussianRecurrent(self.x_dim, mu_init=self.mu_init,
                                                    var_init=self.var_init, corr_init=self.corr_init))

        train_set, test_set = random_split(MyDataset(new_X, new_y, new_task), [0.8, 0.2])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss, test_lkd, test_reg = np.zeros(epoch), np.zeros(epoch), np.zeros(epoch)
        for ep in range(epoch):
            print(ep)
            tl = 0.
            for x, y, task in train_loader:
                optimizer.zero_grad()
                x = x.to(self.task_emb.device)
                loss = torch.tensor(0., dtype=torch.float, device=self.task_emb.device)
                for observed_task_id in torch.unique(task):
                    # compute lkd from curr model parameter
                    id_vec = task == observed_task_id
                    loss -= self.initial_prior_lkd(x=x[id_vec], y=y[id_vec],
                                                   task=observed_task_id, n_context_portion=0.2)/(0.8*x.numel())
                    # regularizing norms of the new task and category embedding
                    loss += embedding_reg * (((self.task_emb[observed_task_id]) ** 2).mean() +
                                             (self.y_emb[observed_task_id] ** 2).mean(1).sum())
                # compute regularization from reference model parameter
                loss += lam * self.continual_regularization(n_pseudo=n_pseudo, prior=True)  # check all previous tasks

                loss.backward()
                optimizer.step()
                tl += loss.detach().cpu().numpy()
            train_loss[ep] = tl/(len(train_set)/len(test_set))

            # work out test loss
            with torch.no_grad():
                tl1, tl2 = 0., 0.
                loss = torch.tensor(0., dtype=torch.float)
                for x, y, task in test_loader:
                    x = x.to(self.task_emb.device)
                    for observed_task_id in torch.unique(task):
                        id_vec = task == observed_task_id
                        loss -= self.initial_prior_lkd(x=x[id_vec], y=y[id_vec],
                                                       task=observed_task_id, n_context_portion=0.2)/(0.8*x.numel())
                    reg = lam * self.continual_regularization(n_pseudo=n_pseudo, prior=True)
                    tl1 += loss.detach().cpu().numpy()
                    tl2 += reg.detach().cpu().numpy()
                test_lkd[ep] = tl1
                test_reg[ep] = tl2

        # after training done, update reference model = curr model, freeze prior layer, update y prior
        self.turn_off_prior_train()  # detach prior layer from graph, treat prior var and cov as fixed
        # update base distributions
        for task_id in new_task_id:
            if len(self.prior_label) != task_id:
                raise ValueError('number of task update goes wrong')
            task_y = new_y[new_task == task_id]
            self.continual_update_task(new_x=new_X[new_task == task_id], new_y=task_y,
                                       task_id=task_id)
            self.prior_label.append(
                torch.tensor([torch.mean(1.0 * (task_y == __)) for __ in range(len(torch.unique(task_y)))],
                             device=self.task_emb.device, requires_grad=False))

        self.update_reference()
        self.task_id_list += new_task_id  # register new task id
        return train_loss, test_lkd, test_reg

    def train_continual_task(self, X_new, y_new, task_id, epoch=100, batch_size=128, lr=1e-3, lam=1., weight_decay=0.,
                             n_pseudo=128, embedding_reg=0.1):
        # Class incremental on a seen task
        n_new_classes = len(set(np.array(torch.unique(y_new))) - set(range(self.y_emb[task_id].shape[0])))
        self.y_emb[task_id] = nn.Parameter(torch.concat((self.reference_y_emb[task_id] * 1.0,
                                                         torch.randn(n_new_classes,
                                                                     self.reference_y_emb[task_id].shape[1]))))

        train_set, test_set = random_split(MyDataset(X_new, y_new, [task_id]*X_new.shape[0]), [0.8, 0.2])
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        train_loss, test_lkd, test_reg = np.zeros(epoch), np.zeros(epoch), np.zeros(epoch)
        for ep in range(epoch):
            print(ep)
            tl = 0.
            for x, y, _ in train_loader:
                optimizer.zero_grad()
                x = x.to(self.task_emb.device)
                # compute lkd from curr model parameter
                loss = -1. * self.task_specific_lkd(x=x, y=y, task=task_id, n_context_portion=0.2)/(0.8*x.numel())
                # compute regularization from reference model parameter
                loss += lam * self.continual_regularization(n_pseudo=n_pseudo)
                # regularizing new class embeddings
                loss += embedding_reg * ((self.y_emb[task_id][self.y_cat_num[task_id]:])**2).mean(1).sum()
                loss.backward()
                optimizer.step()
                tl += loss.detach().cpu().numpy()
            train_loss[ep] = tl/(len(train_set)/len(test_set))

            # work out test loss
            with torch.no_grad():
                tl1, tl2 = 0., 0.
                for x, y, _ in test_loader:
                    x = x.to(self.task_emb.device)
                    loss = -1. * self.task_specific_lkd(x=x, y=y, task=task_id, n_context_portion=0.2)/(0.8*x.numel())
                    reg = lam * self.continual_regularization(n_pseudo=n_pseudo)
                    reg += embedding_reg * ((self.y_emb[task_id][self.y_cat_num[task_id]:])**2).mean(1).sum()
                    tl1 += loss.detach().cpu().numpy()
                    tl2 += reg.detach().cpu().numpy()
                test_lkd[ep] = tl1
                test_reg[ep] = tl2

        # after training done, update reference model = curr model, freeze prior layer, update y prior
        self.continual_update_task(new_x=X_new, new_y=y_new, task_id=task_id)
        self.turn_off_prior_train()
        self.update_reference()  # update all reference models
        # now update prior info, work our portion of new samples
        proportion = torch.tensor(X_new.shape[0]/self.base_list[task_id].sample_size, device=self.task_emb.device)
        # label prior update scheme
        self.y_cat_num[task_id] += n_new_classes
        self.prior_label[task_id] = (torch.tensor([torch.mean(1.0 * (y_new == __)).detach().cpu().item()
                                                   for __ in torch.arange(self.y_cat_num[task_id],
                                                                          device=self.task_emb.device)],
                                                  device=self.task_emb.device) * proportion +
                                     (1-proportion)*torch.cat(
                    (self.prior_label[task_id], torch.zeros(n_new_classes, device=self.task_emb.device))))
        return train_loss, test_lkd, test_reg

    def continual_update_task(self, new_x, new_y, task_id):
        # updating a fully learnt task, only need to update the corresponding Gaussian recurrent layer
        # once updated, freeze parameters and record current parameters as the new prior for this task
        with torch.no_grad():
            task_emb = self.task_emb[task_id].unsqueeze(0).repeat(new_x.shape[0], 1)
            y_emb = self.y_emb[task_id][new_y]
            z, log_det_jac = self.c_nvp(new_x, y_emb, task_emb)
            for i in range(z.shape[0]):
                self.base_list[task_id].update_distribution(z[i])
            self.base_list[task_id].update_restart()

    def prediction(self, x, task_id=None, avg_over_task=False):  # predicting a single observation x
        if task_id is not None:
            with torch.no_grad():
                ans = torch.log(self.prior_label[task_id])  # workout log prior
                z, log_det_jac = self.c_nvp(x.unsqueeze(0).repeat(self.y_emb[task_id].shape[0], 1),
                                            self.y_emb[task_id],
                                            self.task_emb[task_id].unsqueeze(0).repeat(self.y_emb[task_id].shape[0], 1))
                # explore all possible labels under a given task
                for i in range(z.shape[0]):
                    ans[i] += self.base_list[task_id].get_log_likelihood(z[i])  # adding log p(z)
                post = ans + log_det_jac  # adding log det  p(X|y,t)p(y|t)
                lkd_given_task = torch.logsumexp(post, 0)
                post = torch.exp(post - post.max())  # sum exp
                return post / post.sum(), lkd_given_task  # return p(y|X, t) and p(X|t)
        else:  # if no task given, explore all tasks, return a dictionary of cat dists
            task_prior = torch.tensor([_.sample_size for _ in self.base_list],
                                      dtype=torch.float, device=self.task_emb.device).log()
            ans = {}
            marg = torch.zeros(len(task_prior))
            for task_id in self.task_id_list:
                ans[task_id], marg[task_id] = self.prediction(x, task_id=task_id)

            t_given_x = marg + task_prior  # p(X|t)p(t)
            t_given_x = torch.exp(t_given_x - t_given_x.max())
            t_given_x /= t_given_x.sum()  # p(t|X)
            if not avg_over_task:
                return ans, t_given_x
            else:
                y_given_x_t = []
                for _ in self.task_id_list:
                    y_given_x_t.append(ans[_])  # p(y|X,t)

            return (torch.vstack(y_given_x_t) * t_given_x[:, None]).sum(0)

    def turn_off_prior_train(self):
        for prior_layer in self.base_list:
            for params in prior_layer.parameters():
                params.requires_grad = False

    def turn_on_prior_train(self):
        for prior_layer in self.base_list:
            for params in prior_layer.parameters():
                params.requires_grad = True

    def update_reference(self):
        self.reference_model = copy.deepcopy(self.c_nvp)
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.reference_base = copy.deepcopy(self.base_list)
        for prior_layers in self.reference_base:
            for param in prior_layers.parameters():
                param.requires_grad = False

        if self.task_emb is not None:
            self.reference_task_emb = copy.deepcopy(self.task_emb)
            self.reference_task_emb.requires_grad = False

        self.reference_y_emb = copy.deepcopy(self.y_emb)
        for i in self.reference_y_emb:
            i.requires_grad = False


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)