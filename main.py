from __future__ import print_function
import argparse
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

from .sdt_progressive import SDTProgressive


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

    if opt.dataset in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt.dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(opt.imageSize),
                                    transforms.CenterCrop(opt.imageSize),
                                    transforms.ToTensor(),
                                    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    elif opt.dataset == 'lsun':
        dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.CenterCrop(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'cifar10':
        dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                            transform=transforms.Compose([
                                transforms.Resize(opt.imageSize),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                            ]))
    elif opt.dataset == 'fake':
        dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                                transform=transforms.ToTensor())
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                            shuffle=True, num_workers=int(opt.workers))

    opt.fineSize = opt.fineSize or opt.imageSize
    opt.checkpoint_dir = opt.outf

    model = SDTProgressive(opt)
    model.phase = 'train'

    # for epoch in range(opt.niter):
    n_examples = len(dataloader)
    iteration = model.iter_count
    logger = Logger()
    training_start_time = time.time()
    while iteration < opt.niter:
        for data in dataloader:

            real_cpu, _ = data
            model.update(real_cpu)

            iterp1 = iteration + 1
            if iterp1 % opt.train_log_freq == 0:
                errors = model.get_current_errors()
                logger.print_current_errors_csv(iteration, errors)

            if iterp1 % opt.save_freq == 0:
                model.save('%06d' % i)
            if iterp1 % opt.print_iter_freq == 0:
                print('End of iteration %d / %d \t %.3f sec/iter' %
                    (iteration, opt.niter + opt.niter_decay,
                    (time.time() - training_start_time) / (iterp1)))

            model.update_learning_rate()
            iteration += 1


class Logger():
    def __init__(self, directory):
        self.directory = directory
        self.log_dir = directory
        self.img_dir = os.path.join(self.log_dir, 'images')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)

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





if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--fineSize', type=int, default=None, help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=10000, help='number of iterations to train for')
    parser.add_argument('--niter-decay', type=int, default=200, help='number of iterations to decay learning rate')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--lr-policy', type=str, default='lambda', help='lr scheduler policy')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--WGAN-GP_gamma', type=float, default=1.0, help='WGAN-GP gamma')
    parser.add_argument('--WGAN-GP_lambda', type=float, default=1.0, help='WGAN-GP lambda')
    parser.add_argument('--pooling-comp', type=float, default=1.0, help='avg pooling comp')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--auto-continue', action='store_true', help='auto continue training')
    parser.add_argument('--continue-train', action='store_true', help='continue from specified epoch')
    parser.add_argument('--which-epoch', type=int, default=None, help='epoch for continuation of training')
    parser.add_argument('--model-save-freq', type=int, default=100, help='frequency to save model')
    parser.add_argument('--image-save-freq', type=int, default=100, help='frequency to save images')
    parser.add_argument('--comp-const', type=str, default='1/1', help='G/D training period')
    parser.add_argument('--killer', action='store_true')


    opt = parser.parse_args()

    if opt.killer:
        import signal
        import sys

        def signal_handler(signal, frame):
            print('Killing process!')
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    main(opt)
