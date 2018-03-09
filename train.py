from __future__ import print_function
import argparse
import csv
import os
import random
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from progressive_gan import ProgressiveGAN


def main(opt):
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    opt.loadSize = opt.loadSize or opt.imageSize
    opt.fineSize = opt.fineSize or opt.imageSize

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.loadSize),
                                    transforms.CenterCrop(opt.fineSize),
                                    transforms.RandomHorizontalFlip(0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['tower_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.loadSize),
                                transforms.CenterCrop(opt.fineSize),
                                transforms.RandomHorizontalFlip(0.5),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(opt.fineSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, opt.fineSize, opt.fineSize),
                                transform=transforms.ToTensor())
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))

    opt.checkpoint_dir = opt.outf

    model = ProgressiveGAN(opt)
    model.phase = 'train'

    # for epoch in range(opt.niter):
    n_examples = len(dataloader)
    iteration = model.iter_count
    logger = Logger(opt.outf)

    class DataIterator:
        def __init__(self, dataloader):
            self._dataloader = dataloader
            self._iter = None
        def __call__(self):
            tries = 0
            while tries < 2:
                if self._iter is not None:
                    try:
                        data = next(self._iter)
                        return data
                    except StopIteration:
                        pass
                self._iter = iter(self._dataloader)
                tries += 1
            raise RuntimeError('failed to load data from dataloader')
    get_batch = DataIterator(dataloader)

    if opt.start_stage:
        model.iter_count = opt.start_stage * opt.stage_interval / opt.batchSize

    training_start_time = time.time()
    timing_iter = 1
    while model.iter_count < opt.niter + opt.niter_decay:
        for iter_dis in range(opt.n_dis):
            real_cpu, _ = get_batch()
            model.update_discriminator(real_cpu)

        for iter_gen in range(opt.n_gen):
            model.update_generator()

        iteration = model.iter_count
        iterp1 = iteration + 1
        if iterp1 % opt.train_log_freq == 0:
            errors = model.get_current_errors()
            logger.print_current_errors_csv(iteration, errors)
            model.write_summaries()
        if iterp1 % opt.model_save_freq == 0:
            model.save('%06d' % iteration)
        if iterp1 % opt.image_save_freq == 0:
            model.generate_images(scale=True)
            if opt.saliency:
                model.generate_saliency(scale=True)
        if iterp1 % opt.print_iter_freq == 0:
            print('End of iteration %d / %d \t %.3f sec/iter' %
                (iteration, opt.niter + opt.niter_decay,
                (time.time() - training_start_time) / (timing_iter)))

        model.iter_count += 1
        model.update_learning_rate()
        timing_iter += 1


class Logger():
    def __init__(self, directory):
        self.directory = directory
        self.log_dir = directory
        os.makedirs(self.log_dir, exist_ok=True)

        # print('create log directory %s...' % self.log_dir)
        self.log_name_csv = os.path.join(self.log_dir, 'loss_log.csv')

    def print_current_errors_csv(self, iteration, errors):
        write_header = not os.path.exists(self.log_name_csv)

        with open(self.log_name_csv, "a+") as log_file:
            csv_out = csv.writer(log_file)
            if write_header:
                csv_out.writerow(
                    ['iters'] + [k for k, v in errors.items()])
            csv_out.writerow([iteration] + [v for k, v in errors.items()])




def parse_args(default=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train', help='phase of operation')
    parser.add_argument('--dataset', type=str, default='folder', help='cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', type=str, default='dataroot', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=10)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--latent-fixed-bs', type=int, default=None, help='batch size for fixed latent vector used to generate images during training (default None means use batchSize)')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--fineSize', type=int, default=None, help='the height / width of the input image to network')
    parser.add_argument('--loadSize', type=int, default=None, help='the load height / width of the input image')
    parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--conv', type=str, default='impl', help='impl | paper | none | ...')
    parser.add_argument('--linear', type=str, default='impl', help='impl | paper | none | ...')
    parser.add_argument('--niter', type=int, default=10000, help='number of iterations to train for')
    parser.add_argument('--niter-decay', type=int, default=200, help='number of iterations to decay learning rate')
    parser.add_argument('--stage-interval', type=int, default=300000, help='number of examples seen per stage')
    parser.add_argument('--max-stage', type=int, default=6, help='max stage for progressive growth')
    parser.add_argument('--start-stage', type=int, default=None, help='')
    parser.add_argument('--n-gen', type=int, default=1, help='number of generator updates per discriminator')
    parser.add_argument('--n-dis', type=int, default=1, help='number of discriminator updates per discriminator')
    parser.add_argument('--optimizer', type=str, default='adam', help='adam | rms')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lr-policy', type=str, default='lambda', help='lr scheduler policy')
    parser.add_argument('--lr-decay-iters', type=int, default=1e6, help='lr decay steps')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
    parser.add_argument('--WGAN-GP-gamma', type=float, default=1.0, help='WGAN-GP gamma')
    parser.add_argument('--WGAN-GP-lambda', type=float, default=1.0, help='WGAN-GP lambda')
    parser.add_argument('--pooling-comp', type=float, default=1.0, help='avg pooling comp')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='logs', help='folder to output images and model checkpoints')
    parser.add_argument('--name', default=None, help='subfolder within outf for a particular experiment')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--auto-continue', action='store_true', help='auto continue training')
    parser.add_argument('--continue-train', action='store_true', help='continue from specified epoch')
    parser.add_argument('--which-epoch', type=int, default=None, help='epoch for continuation of training')
    parser.add_argument('--train-log-freq', type=int, default=100, help='frequency to log')
    parser.add_argument('--print-iter-freq', type=int, default=100, help='frequency to log')
    parser.add_argument('--model-save-freq', type=int, default=100, help='frequency to save model')
    parser.add_argument('--image-save-freq', type=int, default=100, help='frequency to save images')
    parser.add_argument('--comp-const', type=str, default='1/1', help='G/D training period')
    parser.add_argument('--saliency', action='store_true', help='generate saliency map')
    parser.add_argument('--killer', action='store_true')
    parser.add_argument('--print-network', action='store_true', help='print network structure')

    if default:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    if not (args.n_gen == 1 or args.n_dis == 1):
        raise RuntimeError('invalid combination of n_gen and n_dis -- one must take the value \'1\'')
    if not os.path.isdir(args.dataroot):
        raise RuntimeError('argument \'dataroot\' is not a directory')

    return args


if __name__ == "__main__":
    opt = parse_args()

    if opt.killer:
        import signal
        import sys

        def signal_handler(signal, frame):
            print('Killing process!')
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    main(opt)
