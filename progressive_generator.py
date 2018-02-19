import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

from torch.nn import init
from torch.autograd import Variable, grad


def feature_vector_normalization(x, eps=1e-8):
    # x: (B, C, H, W)
    alpha = 1.0 / torch.sqrt(torch.mean(x*x, dim=1, keepdim=True) + eps)
    return alpha * x


class EqualizedConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, pad):
        super(EqualizedConv2d, self).__init__()
        # w = chainer.initializers.Normal(1.0) # equalized learning rate
        self.inv_c = torch.FloatTensor([np.sqrt(2.0/(in_ch*ksize**2))])
        self.c = nn.Conv2d(in_ch, out_ch, ksize, stride, pad)

        init.normal(self.c.weight.data, 0.0, 1.0)

    def forward(self, x):
        return self.c(Variable(self.inv_c.type_as(x)) * x)

class EqualizedLinear(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(EqualizedLinear, self).__init__()
        self.inv_c = torch.FloatTensor([np.sqrt(2.0 / in_ch)])
        self.c = nn.Linear(in_ch, out_ch)

        init.normal(self.c.weight.data, 0.0, 1.0)
        init.constant(self.c.bias.data, 0.0)
        # !!! TODO Init bias?

    def forward(self, x):
        return self.c(Variable(self.inv_c.type_as(x)) * x)


def minibatch_std(x):
    # m = F.mean(x, axis=0, keepdims=True)
    # v = F.mean((x - F.broadcast_to(m, x.shape))*(x - F.broadcast_to(m, x.shape)), axis=0, keepdims=True)
    # std = F.mean(F.sqrt(v + 1e-8), keepdims=True)
    # std = F.broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
    # return F.concat([x, std], axis=1)

    std = x.std(dim=0, keepdim=True, unbiased=False)
    std = std.mean()
    std = std.expand(x.shape[0], 1, x.shape[2], x.shape[3])
    return torch.cat([x, std], dim=1)


class GenDownBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pooling_comp=1.0):
        super(GenDownBlock, self).__init__()
        self.pooling_comp = pooling_comp
        self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = EqualizedConv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        # x = F.leaky_relu(feature_vector_normalization(self.c0(x)), 0.2)
        x = F.leaky_relu(feature_vector_normalization(self.c0(x)))
        x = F.leaky_relu(feature_vector_normalization(self.c1(x)))
        x = self.pooling_comp * F.avg_pool2d(x, 2, 2, 0)
        return x


class GenUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GenUpBlock, self).__init__()
        self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
        self.c1 = EqualizedConv2d(out_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x = F.upsample(x, scale_factor=2)
        # x = F.leaky_relu(feature_vector_normalization(self.c0(x)), 0.2)
        x = F.leaky_relu(feature_vector_normalization(self.c0(x)))
        x = F.leaky_relu(feature_vector_normalization(self.c1(x)))
        return x


class Generator(nn.Module):
    def __init__(self, nz, out_ch, ngf, max_stage=12,
                 pooling_comp=1.0):
        super(Generator, self).__init__()
        # super().__init__()
        self.max_stage = max_stage
        self.pooling_comp = pooling_comp
        self.nz = nz

        # !!! TODO check indexing from 1 not 0
        self.c0 = GenUpBlock(nz, ngf, 4, 1, 3)
        self.c1 = GenUpBlock(ngf, ngf, 3, 1, 1)
        self.out0 = EqualizedConv2d(ngf, 3, 1, 1, 0)

        self.b1 = GenUpBlock(ngf, ngf)
        self.out1 = EqualizedConv2d(ngf, 3, 1, 1, 0)
        self.b2 = GenUpBlock(ngf, ngf)
        self.out2 = EqualizedConv2d(ngf, 3, 1, 1, 0)
        self.b3 = GenUpBlock(ngf, ngf)
        self.out3 = EqualizedConv2d(ngf, 3, 1, 1, 0)
        self.b4 = GenUpBlock(ngf, ngf)
        self.out4 = EqualizedConv2d(ngf, 3, 1, 1, 0)
        self.b5 = GenUpBlock(ngf, ngf)
        self.out5 = EqualizedConv2d(ngf, 3, 1, 1, 0)
        self.b6 = GenUpBlock(ngf, ngf // 2)
        self.out6 = EqualizedConv2d(ngf // 2, 3, 1, 1, 0)

    def latent(self, batch_size):
        z = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
        z /= z.norm(2) + 1e-8
        return z

    def forward(self, x, stage):
        # int0->db0->  eb0>out0
        # (1-a)*(down->in0) + (a)*(in1->db1) ->db0 ->  eb0> (1-a) * (up->out0) + a*(eb1->out1)
        # in1->db1->db0->eb0->eb1->out1
        # ...
        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        staged2 = int(stage // 2)
        staged2p1 = int(stage // 2 + 1)

        h = z.view(z.shape[0], self.nz, 1, 1)
        h = F.leaky_relu(feature_vector_normalization(self.c0(h)))
        h = F.leaky_relu(feature_vector_normalization(self.c1(h)))

        # Decoder
        for i in range(1, int(staged2p1)):
            h = getattr(self, 'b%d' % i)(h)

        if (int(stage) & 1) == 0:
            out = getattr(self, "out%d" % (staged2))
            x = out(h)
        else:
            out_prev = getattr(self, 'out%d' % (staged2))
            out_curr = getattr(self, 'out%d' % (staged2p1))
            b_curr = getattr(self, 'b%d' % (staged2p1))

            # combined output
            x0 = out_prev(F.upsample(h, scale_factor=2))
            x1 = out_curr(b_curr(h))

            alpha_v = Variable(
                torch.FloatTensor([alpha]).type_as(x),
                requires_grad=False)
            oneminalpha_v = Variable(
                torch.FloatTensor([1. - alpha]).type_as(x),
                requires_grad=False)
            x = oneminalpha_v * x0 + alpha_v * x1

        return x


# class GeneratorParallel(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super(GeneratorParallel, self).__init__()
#         gpu_ids = kwargs.get('gpu_ids', None)
#         if gpu_ids is None:
#             raise RuntimeError('gpu_ids must be a kwarg')
#         self.gpu_ids = gpu_ids
#         self.model = Generator(*args, **kwargs)

#     def forward(self, *args, **kwargs):
#         if 'x' in kwargs:
#             input_data = kwargs['x'].data
#         else:
#             input_data = args[0]

#         print(kwargs)
#         if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
#             return nn.parallel.data_parallel(self.model, args, self.gpu_ids,
#                                              module_kwargs=kwargs)
#         else:
#             return self.model(*args, **kwargs)




class DiscriminatorBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pooling_comp=1.0):
        super(DiscriminatorBlock, self).__init__()
        self.pooling_comp = pooling_comp

        self.c0 = EqualizedConv2d(in_ch, in_ch, 3, 1, 1)
        self.c1 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        h = F.leaky_relu(self.c0(x))
        h = F.leaky_relu(self.c1(h))
        h = self.pooling_comp * F.avg_pool2d(h, 2, 2, 0)
        return h


class Discriminator(nn.Module):
    def __init__(self, in_ch=3, ndf=512, max_stage=12,
                 pooling_comp=1.0):
        super(Discriminator, self).__init__()
        self.max_stage = max_stage
        self.pooling_comp = pooling_comp

        self.in6 = EqualizedConv2d(in_ch, ndf // 2, 1, 1, 0)
        self.b6 = DiscriminatorBlock(ndf // 2, ndf, pooling_comp)
        self.in5 = EqualizedConv2d(in_ch, ndf, 1, 1, 0)
        self.b5 = DiscriminatorBlock(ndf, ndf, pooling_comp)
        self.in4 = EqualizedConv2d(in_ch, ndf, 1, 1, 0)
        self.b4 = DiscriminatorBlock(ndf, ndf, pooling_comp)
        self.in3 = EqualizedConv2d(in_ch, ndf, 1, 1, 0)
        self.b3 = DiscriminatorBlock(ndf, ndf, pooling_comp) # Input Nxndfx32x32
        self.in2 = EqualizedConv2d(in_ch, ndf, 1, 1, 0)
        self.b2 = DiscriminatorBlock(ndf, ndf, pooling_comp)
        self.in1 = EqualizedConv2d(in_ch, ndf, 1, 1, 0)
        self.b1 = DiscriminatorBlock(ndf, ndf, pooling_comp)
        self.in0 = EqualizedConv2d(in_ch, ndf, 1, 1, 0)
        # Differs from generator whindf grows from 0

        # Shape here [Bx4x4x4]
        self.out0 = EqualizedConv2d(ndf + 1, ndf, 3, 1, 1)
        self.out1 = EqualizedConv2d(ndf, ndf, 4, 1, 0)
        self.out2 = EqualizedLinear(ndf, 1)

    def forward(self, x, stage):
        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        staged2 = int(stage // 2)
        staged2p1 = int(stage // 2 + 1)

        if (int(stage) % 2) == 0:
            fromRGB = getattr(self, 'in%d' % (staged2))
            h = F.leaky_relu(fromRGB(x))
        else:
            fromRGB0 = getattr(self, 'in%d' % (staged2))
            fromRGB1 = getattr(self, 'in%d' % (staged2p1))
            b1 = getattr(self, 'b%d' % (staged2p1))

            h0 = F.leaky_relu(fromRGB0(self.pooling_comp * F.avg_pool2d(x, 2, 2, 0)))
            h1 = b1(F.leaky_relu(fromRGB1(x)))

            alpha_v = Variable(
                torch.FloatTensor([alpha]).type_as(x),
                requires_grad=False)
            oneminalpha_v = Variable(
                torch.FloatTensor([1. - alpha]).type_as(x),
                requires_grad=False)

            h = oneminalpha_v * h0 + alpha_v * h1

        # for i in range(staged2, -1, -1):
        for i in range(staged2, 0, -1):
            h = getattr(self, 'b%d' % i)(h)

        h = minibatch_std(h)
        h = F.leaky_relu(self.out0(h))
        h = F.leaky_relu(self.out1(h))
        return self.out2(h.view(*h.shape[:2]))
