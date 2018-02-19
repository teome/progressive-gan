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

    def forward(self, x):
        # x = F.leaky_relu(feature_vector_normalization(self.c0(x)), 0.2)
        x = F.leaky_relu(feature_vector_normalization(self.c0(x)))
        x = self.pooling_comp * F.avg_pool2d(x, 2, 2, 0)
        return x


class GenUpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(GenUpBlock, self).__init__()
        self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)

    def forward(self, x):
        x = F.upsample(x, scale_factor=2)
        # x = F.leaky_relu(feature_vector_normalization(self.c0(x)), 0.2)
        x = F.leaky_relu(feature_vector_normalization(self.c0(x)))
        return x

class Generator(nn.Module):
    def __init__(self, in_ch, out_ch, ngf, max_stage=12,
                 pooling_comp=1.0):
        super(Generator, self).__init__()
        # super().__init__()
        self.max_stage = max_stage
        self.pooling_comp = pooling_comp

        # !!! TODO check indexing from 1 not 0
        self.in0 = EqualizedConv2d(in_ch, ngf * 8, 3, 1, 1)
        self.db0_0 = GenDownBlock(ngf * 8, ngf * 8)
        self.db0_1 = GenDownBlock(ngf * 8, ngf * 8)
        self.ub0_0 = GenUpBlock(ngf * 8, ngf * 8)
        self.ub0_1 = GenUpBlock(ngf * 8, ngf * 8)
        self.out0 = EqualizedConv2d(ngf * 8, out_ch, 1, 1, 0)

        self.in1 = EqualizedConv2d(in_ch, ngf * 8, 3, 1, 1)
        self.db1 = GenDownBlock(ngf * 8, ngf * 8)
        self.ub1 = GenUpBlock(ngf * 8 * 2, ngf * 8)
        self.out1 = EqualizedConv2d(ngf * 8, out_ch, 1, 1, 0)

        self.in2 = EqualizedConv2d(in_ch, ngf * 8, 3, 1, 1)
        self.db2 = GenDownBlock(ngf * 8, ngf * 8)
        self.ub2 = GenUpBlock(ngf * 8 * 2, ngf * 8)
        self.out2 = EqualizedConv2d(ngf * 8, out_ch, 1, 1, 0)

        self.in3 = EqualizedConv2d(in_ch, ngf * 8, 3, 1, 1)
        self.db3 = GenDownBlock(ngf * 8, ngf * 8)
        self.ub3 = GenUpBlock(ngf * 8 * 2, ngf * 8)
        self.out3 = EqualizedConv2d(ngf * 8, out_ch, 1, 1, 0)

        self.in4 = EqualizedConv2d(in_ch, ngf * 4, 3, 1, 1)
        self.db4 = GenDownBlock(ngf * 4, ngf * 8)
        self.ub4 = GenUpBlock(ngf * 8 * 2, ngf * 4)
        self.out4 = EqualizedConv2d(ngf * 4, out_ch, 1, 1, 0)

        self.in5 = EqualizedConv2d(in_ch, ngf * 2, 3, 1, 1)
        self.db5 = GenDownBlock(ngf * 2, ngf * 4)
        self.ub5 = GenUpBlock(ngf * 4 * 2, ngf * 2)
        self.out5 = EqualizedConv2d(ngf * 2, out_ch, 1, 1, 0)

        self.in6 = EqualizedConv2d(in_ch, ngf * 1, 3, 1, 1)
        self.db6 = GenDownBlock(ngf * 1, ngf * 2)
        self.ub6 = GenUpBlock(ngf * 2 * 2, ngf * 1)
        self.out6 = EqualizedConv2d(ngf * 1, out_ch, 1, 1, 0)

        # self.in7 = EqualizedConv2d(in_ch, in_ch, 3, 1, 1)
        # self.db7 = GenDownBlock(in_ch, ngf * 1)
        # self.ub7 = GenUpBlock(ngf * 1 * 2, out_ch)
        # self.out7 = EqualizedConv2d(out_ch, out_ch, 1, 1, 0)

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

        # Encoder
        if (int(stage) & 1) == 0:
            fromRGB0 = getattr(self, 'in%d' % (staged2))
            h = F.leaky_relu(fromRGB0(x))
            # skip_skip = True
        else:
            fromRGB0 = getattr(self, 'in%d' % (staged2))
            fromRGB1 = getattr(self, 'in%d' % (staged2p1))
            b1 = getattr(self, 'db%d' % (staged2p1))

            h0 = F.leaky_relu(
                fromRGB0(self.pooling_comp * F.avg_pool2d(x, 2, 2, 0)))
            h1 = b1(F.leaky_relu(fromRGB1(x)))

            alpha_v = Variable(
                torch.FloatTensor([alpha]).type_as(x),
                requires_grad=False)
            oneminalpha_v = Variable(
                torch.FloatTensor([1. - alpha]).type_as(x),
                requires_grad=False)

            h = oneminalpha_v * h0 + alpha_v * h1
            # skip_skip = False

        skip_cons = []
        # !!! TODO Check this, we don't want to cat skip connection on the last
        # layer, even when growing, but since we have defined the convolutions
        # with the number of channels for out+skip, we probably have to do so
        # until the last
        skip_ix = int(staged2) if (int(stage) & 1 == 0) else int(staged2p1)
        # skip_ix = int(self.max_stage // 2)

        # Begin orig net to 0 with no _0, _1
        # for i in range(int(stage // 2), -1, -1):
        #     if i != skip_ix:
        #         skip_cons.append(h)
        #     h = getattr(self, 'db%d' % i)(h)

        # # Decoder
        # for i in range(0, int(staged2p1)):
        #     h = getattr(self, "ub%d"%i)(h)
        #     if i != skip_ix:
        #         s = skip_cons.pop(-1)
        #         h = torch.cat([h, s], 1)

        # Begin alternate
        for i in range(int(stage // 2), 0, -1):
            if i != skip_ix:
                skip_cons.append(h)
            h = getattr(self, 'db%d' % i)(h)

        # Withoug skip in the middle
        if skip_ix != 0:
            skip_cons.append(h)
        h = self.db0_1(h)
        h = self.db0_0(h)
        h = self.ub0_0(h)
        h = self.ub0_1(h)
        if skip_ix != 0:
            s = skip_cons.pop(-1)
            h = torch.cat([h, s], 1)

        # Using skip connections on
        # db0_1 nothing on db0_0, nothing ub0_0, cat skip ub0_1
        # Requires this for the middle section
        # if skip_ix == 0:
        #     h = self.db0_1(h)
        #     skip_cons.append(h)
        #     h = self.db0_0(h)
        #     h = self.ub0_0(h)
        #     s = skip_cons.pop(-1)
        #     h = torch.cat([h, s], 1)
        #     h = self.ub0_1(h)
        # else:
        #     for i in range(1, -1, -1):
        #         skip_cons.append(h)
        #         h = getattr(self, 'db0_%i' % d)(h)
        #     for i in range(0, 2, 1):
        #         h = getattr(self, 'ub0_%i' % d)(h)
        #         s = skip_cons.pop(-1)
        #         torch.cat([h, s], 1)


        # Decoder
        for i in range(1, int(staged2p1)):
            h = getattr(self, "ub%d"%i)(h)
            if i != skip_ix:
                s = skip_cons.pop(-1)
                h = torch.cat([h, s], 1)



        if (int(stage) & 1) == 0:
            out = getattr(self, "out%d" % (staged2))
            x = out(h)
            # if stage2 != skip_ix:
                # s = skip_cons.pop(-1)
                # x = torch.cat([x, s], 1)
        else:
            out_prev = getattr(self, 'out%d' % (staged2))
            out_curr = getattr(self, 'out%d' % (staged2p1))
            b_curr = getattr(self, "ub%d" % (staged2p1))

            # Don't use h directly from the layer below; it has the skip
            # connection we don't need for this combination with the final
            # growth layer
            # x0 = out_prev(F.upsample(h, scale_factor=2))
            # Slice off the skip connection. We don't need to produce the
            # combined output
            h0 = h[:, :h.shape[1] // 2]
            x0 = out_prev(F.upsample(h0, scale_factor=2))
            x1 = out_curr(b_curr(h))
            # Don't ever use skips for the output
            # if stage2p1 != skip_ix:
                # s = skip_cons.pop(-1)
                # x1 = torch.cat([x1, s], 1)

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
    def __init__(self, in_ch, ngf, max_stage=12,
                 pooling_comp=1.0):
        super(Discriminator, self).__init__()
        self.max_stage = max_stage
        self.pooling_comp = pooling_comp

        # !!! TODO check indexing from 1 not 0
        ch = ngf * 8

        self.in6 = EqualizedConv2d(in_ch, ch // 2, 1, 1, 0)
        self.b6 = DiscriminatorBlock(ch // 2, ch, pooling_comp)

        self.in5 = EqualizedConv2d(in_ch, ch, 1, 1, 0)
        self.b5 = DiscriminatorBlock(ch, ch, pooling_comp)
        self.in4 = EqualizedConv2d(in_ch, ch, 1, 1, 0)
        self.b4 = DiscriminatorBlock(ch, ch, pooling_comp)
        self.in3 = EqualizedConv2d(in_ch, ch, 1, 1, 0)
        self.b3 = DiscriminatorBlock(ch, ch, pooling_comp) # Input Nxchx32x32
        self.in2 = EqualizedConv2d(in_ch, ch, 1, 1, 0)
        self.b2 = DiscriminatorBlock(ch, ch, pooling_comp)
        self.in1 = EqualizedConv2d(in_ch, ch, 1, 1, 0)
        self.b1 = DiscriminatorBlock(ch, ch, pooling_comp)
        self.in0 = EqualizedConv2d(in_ch, ch, 1, 1, 0)
        # Differs from generator which grows from 0
        # self.b0 = DiscriminatorBlock(ch, ch, pooling_comp)

        # Shape here [Bx4x4x4]

        self.out0 = EqualizedConv2d(ch + 1, ch, 3, 1, 1)
        self.out1 = EqualizedConv2d(ch, ch, 4, 1, 0)
        self.out2 = EqualizedLinear(ch, 1)

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
