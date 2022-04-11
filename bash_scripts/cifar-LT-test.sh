export CUDA_VISIBLE_DEVICES=0
port=$[$RANDOM + 10000]

data_folder=${data_folder:-data}

GPU_NUM=${GPU_NUM:-1}
pretrain_name=bcl_res18_cifar100_temp0.2_lr0.5_b512_epoch2000_cifar100_split1_D_i_s10_wd5e-4
# pretrain_name=simclr_res18_cifar100_temp0.2_lr0.5_b512_epoch2000_cifar100_split1_D_i_s10_wd5e-4
checkpoint_pretrain=checkpoints/${pretrain_name}/model_2000.pt

cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} test.py ${pretrain_name} --data_folder ${data_folder} \
--batch_size 1000 --model res18 --seed 10 --dataset cifar100 --num_workers 10 --checkpoint ${checkpoint_pretrain} --test --test_fullshot --test_100shot --test_50shot --test_10shot"

${cmd}