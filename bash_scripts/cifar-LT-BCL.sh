data_folder=${data_folder:-data}

pretrain_split=${pretrain_split:-cifar100_split1_D_i}
pretrain_name=bcl_res18_cifar100_temp0.2_lr0.5_b512_epoch2000_${pretrain_split}_s10_wd5e-4

cmd="python train.py ${pretrain_name} --data_folder ${data_folder} --dataset cifar100 --gpus 0 --seed 10 \
--epochs 2000 --batch_size 512  --num_workers 8 --lr 0.5 --temperature 0.2 --weight_decay 5e-4 --model resnet18 \
--trainSplit cifar100_imbSub_with_subsets/${pretrain_split}.npy --save-dir checkpoints --bcl"

${cmd}