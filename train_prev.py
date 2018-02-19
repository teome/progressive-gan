import time
import torch
from options.train_options import TrainOptions
from data.custom_dataset_data_loader import CreateDataLoader
from models.pix2pix_model import Pix2PixModel
from models.sdt_progressive import SDTProgressive
from util.logger import Logger
from util import html
from util import util
import os
import torch.backends.cudnn as cudnn
import random



def run_validation(model, data, logger):
    model.eval()
    model.set_input(data)
    model.validate()
    model.train()

    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    errors = model.get_current_errors()
    logger.save_images_p2p(visuals, img_path)
    logger.print_current_errors_csv_val(img_path, errors)



def main(opt):
    start_time = time.time()
    cudnn.benchmark = True

    if opt.A_freq:
        train_data_A = CreateDataLoader(
            opt, subset='train', mode='overlayed', dataroot=opt.dataroot_A)
        train_val_A = util.prep_train_val_data(train_data_A['dataset'], opt, 'overlayed')
        print('#train_A: %d' % (len(train_data_A['dataset'])))
        iteration_A = 0
        train_data_A_iter = None

        if opt.validate:
            val_data_A = CreateDataLoader(
                opt, subset='val', mode='overlayed', dataroot=opt.dataroot_A)
            print('#val_A: %d' % len(val_data_A['dataset']))

    if opt.B_freq:
        train_data_B = CreateDataLoader(
            opt, subset='train', mode='aligned', dataroot=opt.dataroot_B)
        train_val_B = util.prep_train_val_data(train_data_B['dataset'], opt, 'aligned')
        print('#train_B: %d' % (len(train_data_B['dataset'])))
        iteration_B = 0
        train_data_B_iter = None

        if opt.validate:
            val_data_B = CreateDataLoader(
                opt, subset='val', mode='aligned', dataroot=opt.dataroot_B)
            print('#val_B: %d' % len(val_data_B['dataset']))

    if opt.which_model == 'pix2pix':
        model = Pix2PixModel(opt)
    else:
        model = SDTProgressive(opt)

    iter_count = model.get_iter_count()

    logger_train = Logger(opt)

    training_start_time = time.time()
    for iteration in range(iter_count, opt.niter + opt.niter_decay + 1):

        if opt.A_freq and (iteration % opt.A_freq == 0):
            iteration_A += 1

            if random.random() < opt.unsup_prob:
                shuffle = True
            else:
                shuffle = False

            data, train_data_A_iter = util.get_next_data(
                train_data_A_iter, train_data_A['data_loader'])
            data = util.unsup_shuffle(data, shuffle, opt, 'overlayed')

            model.set_input(data)
            model.optimize_parameters()

            if iteration % opt.train_save_freq == 0:
                logger = Logger(opt, 'train_A_%04d' % (iteration))
                run_validation(model, train_val_A, logger)

        if opt.B_freq and (iteration % opt.B_freq == 0):
            iteration_B += 1

            shuffle = False
            data, train_data_B_iter = util.get_next_data(
                train_data_B_iter, train_data_B['data_loader'])
            data = util.unsup_shuffle(data, shuffle, opt, 'aligned')

            model.set_input(data)
            model.optimize_parameters()

            if iteration % opt.train_save_freq == 0:
                logger = Logger(opt, 'train_B_%04d' % (iteration))
                run_validation(model, train_val_B, logger)

        if iteration % opt.train_log_freq == 0:
            errors = model.get_current_errors()
            logger_train.print_current_errors_csv(iteration, errors)

        if opt.validate and iteration % opt.val_log_freq == 0:
            print('Validating...')
            if opt.A_freq:
                val_dir = os.path.join('val_A_%04d' % (iteration))
                logger_val = Logger(opt, val_dir)

                for data in val_data_A['data_loader']:
                    shuffle = False
                    data = util.unsup_shuffle(data, shuffle, opt, 'overlayed')
                    run_validation(model, data, logger_val)
            if opt.B_freq:
                val_dir = os.path.join('val_B_%04d' % (iteration))
                logger_val = Logger(opt, val_dir)

                for data in val_data_B['data_loader']:
                    shuffle = False
                    data = util.unsup_shuffle(data, shuffle, opt, 'aligned')
                    run_validation(model, data, logger_val)

        if iteration % opt.save_iter_freq == 0:
            print('saving the model at the end of iteration %d' % (iteration))
            # model.save('latest')
            model.save("%04d" % iteration)

        if iteration % opt.print_iter_freq == 0:
            print('End of iteration %d / %d \t %.3f sec/iter' %
                  (iteration, opt.niter + opt.niter_decay,
                   (time.time() - training_start_time) / (iteration + 1)))
            model.print_learning_rate()
        model.update_learning_rate()
    print("Training done, took %.1f min" % ((time.time() - start_time) / 60.0))


if __name__ == "__main__":

    opt = TrainOptions().parse()

    if opt.killer:
        import signal
        import sys

        def signal_handler(signal, frame):
            print('Killing process!')
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    main(opt)
