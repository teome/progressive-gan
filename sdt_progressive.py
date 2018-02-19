import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision as tv
import torchvision.utils as vutils
import math
from collections import OrderedDict
from torch.nn import init
from torch.autograd import Variable, grad
from torch.optim import lr_scheduler
from . import networks
from .base_model import BaseModel
from .progressive_generator import Generator, Discriminator
import util.util as util


class SDTProgressive(BaseModel):
    def name(self):
        return 'SDTProgressive'

    def __init__(self, opt):
        super(SDTProgressive, self).__init__(opt)

        self.opt = opt
        self.phase = opt.phase
        self.ngpu = opt.ngpu
        self.Tensor = torch.FloatTensor if self.ngpu > 0 else torch.Tensor
        self.outf = opt.outf
        self.batch_size = opt.batchSize
        self.input_nc = 3
        self.max_resolution = opt.fineSize
        self.stage_interval = opt.stage_interval
        self.max_stage = opt.max_stage
        self._stage = None

        size = opt.fineSize
        batch_size = opt.batchSize

        # Note that the base size is 4x4, hence the -2 here
        max_stage = self.resolution_stage(self.max_resolution)
        if max_stage < self.max_stage:
            raise RuntimeError('max stage is too large for the max image resulution %d %d' % (self.max_stage, max_stage))
        print('Progressive max resolution:', self.max_resolution)
        print('Progressive max stage:', self.max_stage)
        print('Progressive resolution for max stage:',
              self.stage_resolution(self.max_stage))

        self.max_stage = max_stage

        netG = Generator(self.nz, self.input_nc, opt.ngf,
                         max_stage=self.max_stage)

        if not opt.phase == 'test':
            netD = Discriminator(self.output_nc, opt.ndf,
                                 max_stage=self.max_stage)
        self.iter_count = self.load_model()

        gpu_ids = range(self.ngpu) if self.ngpu > 0 else None
        self.netG = nn.parallel.DataParallel(netG, device_ids=gpu_ids)
        if not opt.phase == 'test':
            self.netD = nn.parallel.DataParallel(netD, device_ids=gpu_ids)

        # Fixed vector for image generation
        self.real = self.Tensor(batch_size, self.input_nc, size, size)
        self.latent = self.Tensor(batch_size, self.nz, 1, 1)
        self.latent_fixed = self.Tensor(batch_size, self.nz, 1, 1)
        latent_fixed = self.netG.module.sample_latent(self.batch_size)
        self.latent_fixed.copy_(latent_fixed if self.ngpu < 1 else
                                latent_fixed.cuda())
        self.one = self.Tensor([1])
        self.mone = one * -1

        self.old_lr = opt.lr

        if self.phase == 'train':
            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            if opt.optimizer == 'adam':
                self.optimizer_G = torch.optim.Adam(
                    self.netG.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, opt.beta2))
                self.optimizer_D = torch.optim.Adam(
                    self.netD.parameters(),
                    lr=opt.lr,
                    betas=(opt.beta1, opt.beta2))
            elif opt.optimizer == 'rms':
                self.optimizer_G = torch.optim.RMSprop(
                    self.netG.parameters(), lr=opt.lr)
                self.optimizer_D = torch.optim.RMSprop(
                    self.netD.parameters(), lr=opt.lr)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(
                    get_scheduler(optimizer, opt,
                                  self.iter_count))

            self.comparative = 'freq'
            consts = [int(c) for c in opt.comp_const.split("/")]
            self.comp_const = {
                'G': consts[0],
                'D': consts[1],
            }

        print('---------- Networks initialized -------------')
        if opt.print_network:
            networks.print_network(self.netG)
            if not self.phase == 'test':
                networks.print_network(self.netD)
            print('-----------------------------------------------')


    def get_iter_count(self):
        return self.iter_count

    @property
    def stage(self):
        s = float(self.iter_count * self.batch_size) / self.stage_interval
        s_floor = math.floor(s)
        if self._stage is not None:
            if s_floor > self._stage:
                print('Progressive stage %.2f reached' % (s))
        self._stage = s_floor
        return s

    def stage_resolution(self, stage):
        return min(self.max_resolution,
                   4 * 2**((stage // 2)))

    def resolution_stage(self, resol):
        return 2 * (np.log2(resol) - 2)

    def update(self, x_real_cpu):
        stage = self.stage
        max_resol = self.max_resolution

        if self.ngpu > 0:
            x_real_cpu = x_real_cpu.cuda()

        self.input.resize_as_(x_real_cpu).copy_(x_real_cpu)
        x_real = Variable(self.input)

        if (math.floor(stage) & 1) == 0:
            # Fixed stage
            resol = self.stage_resolution(math.floor(stage) + 1)
            scale = max(1, max_resol // resol)
            if scale > 1:
                x_real = F.avg_pool2d(x_real, scale, scale, 0)
        else:
            # Fade stage
            alpha = stage - math.floor(stage)
            resol_low = self.stage_resolution(math.floor(stage))
            resol_high = self.stage_resolution(math.floor(stage) + 1)
            scale_low = max(1, max_resol // resol_low)
            scale_high = max(1, max_resol // resol_high)
            if scale_low > 1:
                x_real_low = F.upsample(
                    F.avg_pool2d(x_real, scale_low, scale_low, 0),
                    size=(resol_high, resol_high))
                x_real_high = F.avg_pool2d(
                    x_real, scale_high, scale_high, 0)
                x_real = (1 - alpha) * x_real_low + alpha * x_real_high

        self.x_real = x_real
        # all inputs must be a tensor for data_parallel
        staget = self.Tensor([stage])

        train_flags = self.should_train()

        if train_flags['D'] and self.phase != 'test':
            self.optimizer_D.zero_grad()

            y_real = self.netD.forward(x_real, stage=staget)
            self.pred_real = y_real
            loss_d_real = y_real.mean()
            loss_d_real.backward(self.mone)

            # Fake samples
            z = self.netG.sample_latent(x_real.shape[0])
            z = Variable(z)
            x_fake = self.netG(z, stage=staget)
            self.x_fake = x_fake
            y_fake = self.netD(x_fake.detach(), stage=staget)
            self.pred_fake = y_fake
            loss_d_fake = y_fake.mean()
            loss_d_fake.backward(self.one)
            loss_d = loss_d_fake - loss_d_real

            if self.phase == 'train' or self.phase == 'eval':
                eps = torch.rand([x_real.size(0)] + [1] * (x_real.dim() - 1)).type_as(x_real.data)

                x_interp = eps * x_real.data + (1.0 - eps) * x_fake.data
                x_interp = Variable(x_interp, requires_grad=True)
                y_interp = self.netD(x_interp, stage=staget)

                dy_interp = grad(
                        outputs=y_interp,
                        inputs=x_interp,
                        grad_outputs=torch.ones(y_interp.size()) if self.ngpu < 1 else torch.ones(y_interp.size()).cuda(),
                        create_graph=True,
                        retain_graph=True,
                        only_inputs=True)[0]
                dy_interp = dy_interp.view(dy_interp.size(0), -1)

                dy_interp = dy_interp.norm(2, dim=1)
                # Two-sided gradient penalty
                gamma = self.opt.WGAN_GP_gamma
                penalty = F.mse_loss(
                    dy_interp,
                    Variable(gamma * torch.ones_like(dy_interp.data).type_as(dy_interp.data))
                    ) * (1.0 / gamma**2)
                loss_d_gp = penalty
            else:
                loss_d_gp = Variable(self.Tensor([0]))


            # gradient penalty plus loss for parameter drift
            loss_d_penalty = loss_d_gp + 0.001 * (y_real * y_real).mean()
            loss_d_penalty.backward()
            loss_d += loss_d_penalty

            if self.phase == 'train':
                # self.loss_D.backward()
                # loss_d.backward()
                self.optimizer_D.step()

            log_losses = dict(
                loss_d=loss_d.data[0],
                loss_d_real=loss_d_real.data[0],
                loss_d_fake=loss_d_fake.data[0],
                loss_d_gp=loss_d_gp.data[0]
            )
        else:
            log_losses = dict(
                loss_d=0.,
                loss_d_real=0.,
                loss_d_fake=0.,
                loss_d_gp=0.,
            )

        if train_flags['G']:
            self.optimizer_G.zero_grad()
            # Fake samples
            z = self.netG.sample_latent(x_real.shape[0])
            z = Variable(z)
            x_fake = self.netG(z, stage=staget)
            self.x_fake = x_fake
            y_fake = self.netD(x_fake, stage=staget)
            loss_g = -y_fake.mean()
            loss_g.backward()

            if self.phase == 'train':
                # self.loss_G.backward()
                # loss_g.backward()
                self.optimizer_G.step()

            log_losses.update({'loss_g':loss_g.data[0]})
        else:
            log_losses.update({'loss_g':0.0})
        self.log_losses = log_losses

        if self.phase == 'train':
            self.iter_count += 1

    def optimize_parameters(self):
        self.update()

    def validate(self):
        old_iter_count = self.iter_count
        old_phase = self.phase
        self.phase = 'val'

        with torch.no_grad():
            self.update()

        self.phase = old_phase

    def eval_model(self):
        old_phase = self.phase
        self.phase = 'eval'
        self.update()
        self.phase = old_phase

    # no backprop gradients
    def test(self):
        raise NotImplementedError
        with torch.no_grad():
            self.real_A = Variable(self.input_A)
            self.fake_B = self.netG.forward(self.real_A)
            self.real_B = Variable(self.input_B)

    def get_image_paths(self):
        return self.image_paths

    def _load_model(self, epoch):
        epoch_label = "%06d" % (epoch)
        self.load_network(self.netG, 'G', epoch_label)
        if not self.phase == 'test':
            self.load_network(self.netD, 'D', epoch_label)

    def load_model(self):

        latest_iter = 0

        if self.opt.auto_continue and not self.opt.continue_train and self.phase == 'train':
            last_iters = get_latest_checkpoints(self.opt)
            if last_iters:
                for iteration in last_iters:
                    try:
                        print('Trying to load iter %d.' % iteration)
                        self._load_model(iteration)
                        latest_iter = iteration
                        print('Auto reload from iter %d successful' %
                              latest_iter)
                        break
                    except Exception as e:
                        print(e)
                        print(
                            'Model load failed from epoch %d.' % iteration)
                if latest_iter == 0:
                    print('All model load failed, starting from 0.')
            else:
                print('Auto reload failed, no checkpoint found')

        elif self.opt.continue_train or not self.phase == 'train':
            iteration = self.opt.which_iter
            try:
                self._load_model(iteration)
                latest_iter = iteration
                print('Reload from epoch %d successful' % latest_iter)
            except Exception as e:
                print(e)
                print('Model load failed from epoch %d.' % iteration)

        iter_count = latest_iter
        return iter_count

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if type(network) == torch.nn.parallel.DataParallel:
            devs = network.device_ids
            net_module = network.module
            torch.save(net_module.cpu().state_dict(), save_path)
            if len(devs) == 1:
                net_module.cuda(devs[0])
        else:
            torch.save(network.cpu().state_dict(), save_path)
            if self.ngpu > 0 and torch.cuda.is_available():
                network.cuda(device=0)

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if type(network) == torch.nn.parallel.DataParallel:
            network.module.load_state_dict(torch.load(save_path))
        else:
            network.load_state_dict(torch.load(save_path))

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def print_learning_rate(self):
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def eval(self):
        self.netG.eval()
        if not self.phase == 'test':
            self.netD.eval()

    def train(self):
        self.netG.train()
        if not self.phase == 'test':
            self.netD.train()

    def get_current_errors(self):
        return OrderedDict([
            ('G', self.log_losses['loss_g']),
            ('D_real', self.log_losses['loss_d_real']),
            ('D_fake', self.log_losses['loss_d_fake']),
            ('D_GP', self.log_losses['loss_d_gp']),
            ('D', self.log_losses['loss_d']),
        ])

    def generate_images(self, n=None):
        n = n or self.latent_fixed.shape[0]
        vutils.save_image(self.input.data[:n],
                '%s/real_samples.png' % self.opt.outf,
                normalize=True)
        fake = self.netG.forward(self.latent_fixed[:n],
                                 stage=self.Tensor(self.stage))
        vutils.save_image(fake.data,
                '%s/fake_samples_epoch_%06d.png' % (opt.outf, self.iter_count),
                normalize=True)

    def get_current_visuals(self):
        visuals = []
        for b in range(self.real_A.size()[0]):
            real_A = util.tensor2im(self.real_A[b].data)
            fake_B = util.tensor2im(self.fake_B[b].data)
            real_B = util.tensor2im(self.real_B[b].data)
            visuals.append(
                OrderedDict([('real_A', real_A), ('fake_B', fake_B),
                             ('real_B', real_B)]))
        return visuals

    def save(self, label):
        self.save_network(self.netG, 'G', label)
        self.save_network(self.netD, 'D', label)

    def should_train(self):
        """Return a list (G, D_A, D_B) of flags, e.g. [0, 1, 0] means only D_A
        should be trained."""

        flags = {}
        for key, const in self.comp_const.items():
            if self.comparative == "freq":
                flags.update({key: (self.iter_count % const == 0)})
            else:
                flags.update({key: True})

        return flags


def get_lambda_rule(opt, iter_count):
    lambda_rule = lambda iteration: 1.0 - max(0, iteration + 1 + iter_count - opt.niter) / float(opt.niter_decay + 1)
    return lambda_rule


def get_scheduler(optimizer, opt, iter_count=1):
    if opt.lr_policy == 'lambda':

        lambda_rule = get_lambda_rule(opt, iter_count)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def get_latest_checkpoints(opt):
    chkp_suffix = '_net_D.pth'
    chkp_pattern = os.path.join(opt.checkpoints_dir, opt.name,
                                '*%s' % chkp_suffix)
    list_of_files = glob.glob(chkp_pattern)

    if len(list_of_files) == 0:
        latest_epochs = None
    else:
        latest_files = sorted(
            list_of_files, key=os.path.getctime, reverse=True)[:2]
        latest_epochs = [
            int(os.path.basename(lf).strip(chkp_suffix)) for lf in latest_files
        ]

    return latest_epochs

