# ds_root='/home/mval/datasets/cyclegan_celeb/origs/street_grey_3/'
dt_dir='/scratch/progressive_0'
ds_root=$dt_dir'/lsun'
chkp_dir='/home/mval/data/progressive_1/logs_2/'
proj_dir=$dt_dir'/code'
machine='mval5.mval'
max_stage=8
stage_interval=800000
stage_iters=$(($stage_interval/16))
iters=$((${stage_iters}*14))
model_save_freq=$((${stage_iters}/20)) # 1175
image_save_freq=$(($stage_iters/100))
print_iter_freq=$(($stage_iters/200))
train_log_freq=$print_iter_freq

# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 2 --killer --niter $iters --batchSize 64 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --outf $chkp_dir --dataroot $ds_root --lr 0.0002 --beta1 0.5 --ngpu 2 --beta2 0.999 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 2 --killer --niter $iters --batchSize 64 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --outf $chkp_dir --dataroot $ds_root --lr 0.00002 --beta1 0.5 --ngpu 2 --beta2 0.999 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq

# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 64 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataroot $ds_root --lr 0.00002 --beta1 0.5 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 64 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataroot $ds_root --lr 0.0004 --beta1 0.5 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 64 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataroot $ds_root --lr 0.00004 --beta1 0.5 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq

# 20180226
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 64 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataroot $ds_root --lr 0.0002 --beta1 0.5 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 64 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataroot $ds_root --lr 0.00002 --beta1 0.5 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 64 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataroot $ds_root --lr 0.0004 --beta1 0.5 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 64 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataroot $ds_root --lr 0.00004 --beta1 0.5 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10 

# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.001 --beta1 0.0 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 1
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.0001 --beta1 0.0 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 1
sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.00005 --beta1 0.0 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 10
# sleep 1.1; python cv_condor.py --project-dir $proj_dir --checkpoints-dir $chkp_dir --machine $machine --auto-continue --num-gpus 4 --killer --niter $iters --batchSize 16 --niter-decay 0 --fineSize 64 --loadSize 64 --workers 10 --dataset lsun --dataroot $ds_root --lr 0.0005 --beta1 0.0 --ngpu 4 --beta2 0.99 --max-stage $max_stage --stage-interval $stage_interval --print-iter-freq $print_iter_freq --train-log-freq $print_iter_freq --model-save-freq $model_save_freq --image-save-freq $image_save_freq --WGAN-GP-lambda 1









# export NV_GPU=$CUDA_VISIBLE_DEVICES
# cat /etc/hostname
# exec /usr/bin/nvidia-docker run --rm --ipc=host --sig-proxy=true --volume /scratch/progressive_0/code:/scratch/progressive_0/code --volume /home/mval:/home/mval --volume /scratch:/scratch --workdir /scratch/progressive_0/code --user 1002:999 vj/pytorch:gan python /scratch/progressive_0/code/train.py --auto-continue --batchSize 64 --beta1 0.5 --beta2 0.99 --outf /home/mval/data/progressive_0/logs_0/set_0 --dataroot /scratch/progressive_0/data/ --fineSize 64 --killer --loadSize 64 --lr 0.0002 --manualSeed 0 --max-stage 8 --name model_20180221-210000_0 --niter 47000 --niter-decay 0 --ngpu 1 --phase train --print-iter-freq 47 --train-log-freq 47 --model-save-freq 235 --image-save-freq 235 --stage-interval 300800
