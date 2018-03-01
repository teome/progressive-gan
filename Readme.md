# Desc

Current training for WGAN-GP lambda comparison.
Using a value of 1 for this hyperparam allows the generator weights to diverge too much.
This seems to become more necessary when there is no other form of regularisation on the weights.

Currently training with lambda = 10 for 
/home/dak2/mvaldata/data/progressive_1/logs_2/model_20180226-232527_0/images
/home/dak2/mvaldata/data/progressive_1/logs_2/model_20180227-104958_0/images
/home/dak2/progressive-logs/model_20180227_100000_0/images

And using lambda = 1 for a model with started training at stage 7
/home/dak2/mvaldata/data/progressive_1/logs_2/model_20180227-112233_0/images

-- Schedd: mval3.mval : <[fd00:4d56:414c:8101::3:1]:13108?... @ 02/28/18 21:57:14
 ID      OWNER            SUBMITTED     RUN_TIME ST PRI SIZE CMD
8355.0   dak2            2/25 20:11   0+08:59:34 H  0    8.0 bash /home/mval/data/progressive_1/logs_2/model_20180225-201154_0/docker_run
8357.0   dak2            2/25 20:11   0+08:24:32 H  0    5.0 bash /home/mval/data/progressive_1/logs_2/model_20180225-201157_0/docker_run
8366.0   dak2            2/27 10:14   0+00:52:43 H  0    8.0 bash /home/mval/data/progressive_1/logs_2/model_20180227-101405_0/docker_run
8371.0   dak2            2/27 11:22   0+10:20:49 H  0    8.0 bash /home/mval/data/progressive_1/logs_2/model_20180227-112233_0/docker_run
8377.0   dak2            2/27 13:51   0+07:51:42 H  0    8.0 bash /home/mval/data/progressive_1/logs_2/model_20180227-104958_0/docker_run
8381.0   dak2            2/27 21:46   1+00:10:17 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214656_0/docker_run
8383.0   dak2            2/27 21:47   1+00:09:57 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214700_0/docker_run
8384.0   dak2            2/27 21:47   1+00:09:57 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214702_0/docker_run
8385.0   dak2            2/27 21:47   1+00:09:57 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214704_0/docker_run
8386.0   dak2            2/27 21:47   1+00:09:57 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214706_0/docker_run
8387.0   dak2            2/27 21:47   1+00:09:57 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214709_0/docker_run
8388.0   dak2            2/27 21:48   1+00:08:37 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214827_0/docker_run
8389.0   dak2            2/27 22:02   0+23:54:17 R  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220247_0/docker_run
8390.0   dak2            2/27 22:02   0+23:54:17 R  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220249_0/docker_run
8391.0   dak2            2/27 22:02   0+23:54:17 R  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220251_0/docker_run
8392.0   dak2            2/27 22:02   0+23:54:17 R  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220253_0/docker_run
8393.0   dak2            2/27 22:02   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220255_0/docker_run
8394.0   dak2            2/27 22:02   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220257_0/docker_run
8395.0   dak2            2/27 22:02   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220259_0/docker_run
8396.0   dak2            2/27 22:08   0+23:48:37 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220825_0/docker_run
8397.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220827_0/docker_run
8398.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220829_0/docker_run
8399.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220831_0/docker_run
8400.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220833_0/docker_run
8401.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220835_0/docker_run
8419.0   dak2            2/28 14:10   0+07:46:36 R  0    5.0 bash /home/mval/data/vincent-train-images/model_20180228-141038_0/docker_run



## Lower learning rate and higher discriminator rate
# 20180228: lower learning rate, higher discriminator freq for some
# mval4 higher disc freq
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 1 --num-cpus 8 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.00001 --beta1 0.0 --ngpu 1 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 --n-dis 2
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 1 --num-cpus 8 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.00001 --beta1 0.0 --ngpu 1 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 --n-dis 1
# mval5 mixed lr and dis freq
sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 1 --num-cpus 8 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.00001 --beta1 0.0 --ngpu 1 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 --n-dis 1
sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 1 --num-cpus 8 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.00001 --beta1 0.0 --ngpu 1 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 --n-dis 2
sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 1 --num-cpus 8 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.000005 --beta1 0.0 --ngpu 1 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 --n-dis 1
sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 1 --num-cpus 8 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.00010 --beta1 0.0 --ngpu 1 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 --n-dis 5

-- Schedd: mval3.mval : <[fd00:4d56:414c:8101::3:1]:13108?... @ 03/01/18 09:53:12
 ID      OWNER            SUBMITTED     RUN_TIME ST PRI SIZE CMD
8355.0   dak2            2/25 20:11   0+08:59:34 H  0    8.0 bash /home/mval/data/progressive_1/logs_2/model_20180225-201154_0/docker_run
8357.0   dak2            2/25 20:11   0+08:24:32 H  0    5.0 bash /home/mval/data/progressive_1/logs_2/model_20180225-201157_0/docker_run
8366.0   dak2            2/27 10:14   0+00:52:43 H  0    8.0 bash /home/mval/data/progressive_1/logs_2/model_20180227-101405_0/docker_run
8371.0   dak2            2/27 11:22   0+10:20:49 H  0    8.0 bash /home/mval/data/progressive_1/logs_2/model_20180227-112233_0/docker_run
8377.0   dak2            2/27 13:51   0+07:51:42 H  0    8.0 bash /home/mval/data/progressive_1/logs_2/model_20180227-104958_0/docker_run
8381.0   dak2            2/27 21:46   1+12:06:15 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214656_0/docker_run o 
8384.0   dak2            2/27 21:47   1+12:05:55 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214702_0/docker_run x
8391.0   dak2            2/27 22:02   1+00:09:35 H  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220251_0/docker_run
8392.0   dak2            2/27 22:02   1+00:09:35 H  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220253_0/docker_run
8393.0   dak2            2/27 22:02   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_4/model_20180227-220255_0/docker_run
8397.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220827_0/docker_run
8398.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220829_0/docker_run
8399.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220831_0/docker_run
8400.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220833_0/docker_run
8401.0   dak2            2/27 22:08   0+00:00:00 H  0    1.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-220835_0/docker_run
8419.0   dak2            2/28 14:10   0+19:42:34 R  0    5.0 bash /home/mval/data/vincent-train-images/model_20180228-141038_0/docker_run
8420.0   dak2            2/28 22:01   0+11:49:40 R  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180228-220116_0/docker_run x
8421.0   dak2            2/28 22:01   0+11:39:40 R  0   10.0 bash /home/mval/data/progressive_1/logs_4/model_20180228-220118_0/docker_run x
8426.0   dak2            2/28 22:28   0+11:24:35 R  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180228-222837_0/docker_run o
8427.0   dak2            2/28 22:28   0+11:24:15 R  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180228-222839_0/docker_run o
8428.0   dak2            2/28 22:28   0+11:24:15 R  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180228-222841_0/docker_run o
8429.0   dak2            2/28 22:28   0+11:24:15 R  0    8.0 bash /home/mval/data/progressive_1/logs_4/model_20180228-222843_0/docker_run o

8381.0   dak2            2/27 21:46   1+12:06:15 R  0    8.0 bash /home/mval/data/progressive_1/logs_3/model_20180227-214656_0/docker_run o # sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 1 --num-cpus 8 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.00001 --beta1 0.0 --ngpu 1 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10

## 128 resolution
8435.0   dak2            3/1  15:48   0+00:00:00 I  0    1.0 bash /home/mval/data/progressive_1/logs_5/model_20180301-154827_0/docker_run ? sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 2 --num-cpus 8 --killer --niter $iters --batchSize 32 --niter-decay 0 --fineSize 128 --loadSize 128 --workers 6 --dataset lsun --dataroot $ds_root --lr 0.00001 --beta1 0.0 --ngpu 2 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 --n-dis 1
8436.0   dak2            3/1  15:48   0+00:00:00 I  0    1.0 bash /home/mval/data/progressive_1/logs_5/model_20180301-154829_0/docker_run ? sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 2 --num-cpus 8 --killer --niter $iters --batchSize 32 --niter-decay 0 --fineSize 128 --loadSize 128 --workers 6 --dataset lsun --dataroot $ds_root --lr 0.00001 --beta1 0.0 --ngpu 2 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 --n-dis 2
