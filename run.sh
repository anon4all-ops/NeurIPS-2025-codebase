python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.1 \
    --batch_size 256 \
    --weight_decay 3.5e-3  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_Clean \
    --attack Clean \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0 


python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.1 \
    --batch_size 256 \
    --weight_decay 3.5e-3  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_Corruption \
    --attack Corruption \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0 




python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.01 \
    --batch_size 256 \
    --weight_decay 3.5e-3  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_CVaR \
    --attack CVaR \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0 \
    --cvar 20



python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.01 \
    --batch_size 256 \
    --weight_decay 3.5e-3  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_PGD \
    --attack PGD \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0 


python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.01 \
    --batch_size 256 \
    --weight_decay 3.5e-3  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_TRADES \
    --attack TRADES \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0 


python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.01 \
    --batch_size 256 \
    --weight_decay 3.5e-3  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_MART \
    --attack MART \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0 


python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.01 \
    --batch_size 256 \
    --weight_decay 3.5e-3  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_KL \
    --attack KL \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0



python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.01 \
    --batch_size 256 \
    --weight_decay 3.5e-3  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_ALP \
    --attack ALP \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0


python main.py \
    --dataset CIFAR10 \
    --data_root ./dataset/cifar_10 \
    --model_name resnet18 \
    --input_size 32 \
    --model_depth 28 \
    --model_width 10 \
    --num_class 10 \
    --lr 0.01 \
    --batch_size 256 \
    --weight_decay 3.5e-3  \
    --epochs 100 \
    --save_path new_out/cifar10_res18/AT_CLP \
    --attack CLP \
    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0

    --attack_steps 10 \
    --attack_eps 8.0 \
    --attack_lr 2 \
    --phase train \
    --beta 6.0 
