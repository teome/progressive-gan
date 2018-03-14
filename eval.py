import argparse
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torchvision.utils as vutils
from torch.autograd import Variable

# from progressive_gan import ProgressiveGAN
from progressive_generator import Generator

def main(opt):
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    os.makedirs(opt.outf, exist_ok=True)

    netG = Generator(opt.nz, opt.ngf, opt.max_stage,
                     opt.pooling_comp, opt.conv)

    if os.path.isdir(opt.netG):
        assert os.path.exists(opt.netG), 'netG does not exist'
        nets = list(filter(os.listdir(opt.netG),
                           lambda s: 'netG' in s and '.pth' in s))
        fname = sorted(nets)[-1]
    else:
        fname = opt.netG
    netG.load_state_dict(torch.load(netG))

    # Make sure it's not saved as DataParallel by mistake
    if isinstance(netG, torch.nn.parallel.DataParallel) and opt.ngpu < 2:
        netG = netG.module
    elif opt.ngpu > 1:
        netG = torch.nn.parallel.DataParallel(netG.cuda(), list(opt.ngpu))
    elif opt.ngpu == 1:
        netG = netG.cuda()
    use_cuda = opt.ngpu > 0
    Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    batch_size = opt.batchSize
    step = 1. / opt.steps
    z0 = netG.sample_latent(batch_size)
    for iter in range(opt.niter):
        print('Begin iter %04d' % iter)
        z1 = netG.sample_latent(batch_size)
        if use_cuda:
            z0 = z0.cuda()
            z1 = z1.cuda()

        beta = 0.
        for i in range(opt.steps):
            print('t\step %06d' % i)
            alpha = 1. - beta
            alpha_v = Variable(Tensor([alpha]))
            beta_v = Variable(Tensor([beta]))

            z = alpha_v * z0 + beta_v * z1
            z /= torch.sqrt(torch.mean(z * z))
            x = netG(z, opt.max_stage)

            vutils.save_image(
                self.x,
                os.path.join(opt.outf,
                             'interp_%04d_%06d.png' % (iter, i)),
                normalize=True,
                scale_each=True)


            beta += step

        z0 = z1.clone()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--niter', type=int, default=1, help='number of iterations  for interpolation')
    parser.add_argument('--steps', type=int, default=64, help='number of steps per iteration')
    parser.add_argument('--max-stage', type=int, default=6, help='max stage for progressive growth')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--conv', type=str, default='impl', help='impl | paper | none | ...')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--outf', default='interp', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--no-killer', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if not opt.no_killer:
        import signal
        import sys

        def signal_handler(signal, frame):
            print('Killing process!')
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


    main(args)

