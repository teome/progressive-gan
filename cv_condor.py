import os
import glob
import shutil
import subprocess
import random
from time import gmtime, strftime
from collections import namedtuple
import json
import argparse
import numpy as np
import tempfile

from options.submit_options import SubmitTrainOptions
opt = SubmitTrainOptions()

args = opt.parse()
args_dict = vars(args)


def tostr(v):
    if isinstance(v, str):
        return v
    elif isinstance(v, int):
        return str(v)
    elif isinstance(v, bool):
        return str(v)
    elif isinstance(v, float):
        return str(np.round(v, 10))
    elif isinstance(v, list):
        return ' '.join(str(x) for x in v)
    else:
        raise TypeError('invalid type for argument: ' + str(v))


def run_training(test_num, train_args):

    now = strftime("%Y%m%d-%H%M%S", gmtime())
    if args.name == 'experiment_name':
        name = "model_%s_%s" % (str(now), str(test_num))
        train_args.update({"name": name})
    else:
        train_args.update({"name": args.name})

    log_dir = os.path.join(args.project_dir, args.checkpoints_dir,
                           train_args['name'])
    print("Log_dir: " + log_dir)
    p_total = ''
    for pi in os.path.split(log_dir):
        p_total = os.path.join(p_total, pi)
        if not os.path.isdir(p_total):
            os.mkdir(p_total)

    with open(os.path.join(log_dir, 'readme.txt'), 'a') as f:
        f.write('%s\n' % (args.readme))

    ksorted = sorted(list(train_args.keys()))

    with open(os.path.join(log_dir, 'submit_args.json'), 'w') as fp:
        json.dump(args_dict, fp)

    uid = subprocess.check_output(['id', '-u']).decode('utf-8').strip('\n')
    docker_command = (
        'export NV_GPU=$CUDA_VISIBLE_DEVICES\n' +
        'cat /etc/hostname\n' +
        'exec /usr/bin/nvidia-docker run --rm --ipc=host --sig-proxy=true ' +
        '--volume %s:%s ' % (args.project_dir, args.project_dir) +
        '--volume %s:%s ' % ('/home/mval', '/home/mval') +
        '--volume %s:%s ' % ('/scratch', '/scratch') +
        '--workdir %s ' % (args.project_dir) +
        '--user %s:%s ' % (uid, '999') +
        'vj/pytorch:gan python ' + os.path.join(
            args.project_dir, 'train.py') + ' ')
    docker_command += ' '.join('--%s %s' % (k.replace('_', '-'),
                                            tostr(train_args[k]))
                               for k in ksorted if train_args[k] is not None) + '\n'
    # docker_command = '/bin/bash -c env'

    condor_submit_file = os.path.join(log_dir, 'condor_submit_desc')
    docker_run_file = os.path.join(log_dir, 'docker_run')

    cond_args = 'Arguments = %s' % (docker_run_file)
    if args.machine == 'mval5.mval':
        num_cpus = 10
        memory = 40
        machines_req = "(machine == \"mval5.mval\")"
    elif args.machine == 'any':
        num_cpus = 8
        memory = 25
        machines_req = "((machine == \"mval1.mval\") || (machine == \"mval20.mval\") || (machine == \"mval3.mval\") || (machine == \"mval4.mval\") || (machine == \"mval5.mval\"))"
    else:
        num_cpus = 4
        memory = 20
        machines_req = "(machine == \"%s\")" % args.machine

    cond_desc = """\
    Executable   = /bin/bash
    Universe     = vanilla
    Environment = "PATH=/usr/bin:/bin"

    Requirements = {machines_req}
    Request_cpus = {cpus}
    Request_gpus = {gpus}
    Request_memory = {memory}*1024
    logdir = {log_dir}
    Output  = $(logdir)/out.$(process)
    Error  = $(logdir)/err.$(process)
    Log  = $(logdir)/log.$(process)

    {arguments}
    Queue
    """.format(
        machines_req=machines_req,
        cpus=args.num_cpus or num_cpus,
        gpus=args.num_gpus,
        memory=args.memory or memory,
        log_dir=log_dir,
        arguments=cond_args,
    )
    print(cond_desc)

    with open(condor_submit_file, 'w') as tfile:
        tfile.write(cond_desc)
    with open(docker_run_file, 'w') as tfile:
        tfile.write(docker_command)

    subprocess.check_call(['condor_submit', condor_submit_file])

    with open('./run.log', 'a') as f:
        f.write('%s %s\n' % (now, ' '.join(
            ('--{} {}'.format(k, v) for k, v in args._get_kwargs()))))


def run_cv():

    sweep_params = dict(
        lr=[random.uniform, 0.00002, 0.002, args.max_runs],
        identity=[random.uniform, 0.0, 1.0, args.max_runs],
        lambda_A=[random.uniform, 1.0, 100.0, args.max_runs],
        lambda_B=[random.uniform, 1.0, 100.0, args.max_runs],
        beta1=[random.uniform, 0.05, 0.99999, args.max_runs],
    )

    train_args = ({
        k.replace('-', '_'): args_dict[k.replace('-', '_')]
        for k in [key for key in opt.train_args] +
        [key for key in opt.base_args] + list(sweep_params.keys())
        if k.replace('-', '_') in args_dict
    })

    for i in range(args.max_runs):
        if args.rand:
            for k, p in sweep_params.items():
                args[k] = p[0](*p[1:-1])

        if args.sweep is not None:
            assert args.sweep in sweep_params.keys()
            p = sweep_params[args.sweep]
            train_args[args.sweep] = np.linspace(*p[1:])[i]

        run_training(i, train_args)


if __name__ == '__main__':
    run_cv()
