export CUDA_VISIBLE_DEVICES=0
port=$[$RANDOM + 10000]

data_folder=${data_folder:-data}

GPU_NUM=${GPU_NUM:-1}
pretrain_split=${pretrain_split:-cifar100_split1_D_i}
pretrain_name=simclr_res18_cifar100_temp0.2_lr0.5_b512_epoch2000_${pretrain_split}_s10_wd5e-4

cmd="python -m torch.distributed.launch --nproc_per_node=${GPU_NUM} --master_port ${port} train.py ${pretrain_name} --epochs 2000 \
--batch_size 512 --optimizer sgd --lr 0.5 --temperature 0.2 --model res18 --weight_decay 5e-4 \
--trainSplit cifar100_imbSub_with_subsets/${pretrain_split}.npy --save-dir checkpoints --seed 10 \
--dataset cifar100 --num_workers 8 --test_10shot --test_50shot --test_100shot --data_folder ${data_folder}"

${cmd}