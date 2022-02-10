echo "Read config file: $1";
mode=$(grep -A0 'MODEL_NAME:' $1 | cut -c 13- | cut -d# -f1 | tr -d '"' | sed "s/ //g"); echo "$mode"
if [ "$mode" = "LSeSim" ]; then
    exp=$(grep -A0 'EXPERIMENT_NAME:' $1 | cut -c 18- | tr -d '"')
    data=$(grep -A0 'TRAIN_ROOT:' $1 | cut -c 15- | tr -d '"')
    augment=$(grep -A0 'Augment:' $1 | cut -c 12- | cut -d# -f1 | sed "s/ //g")
    if [ "$augment" = "True" ]; then
        set -ex
        python train.py  \
        --dataroot $data \
        --name $exp \
        --model sc \
        --gpu_ids 1 \
        --lambda_spatial 10 \
        --lambda_gradient 0 \
        --attn_layers 4,7,9 \
        --loss_mode cos \
        --gan_mode lsgan \
        --display_port 8093 \
        --direction AtoB \
        --patch_size 64 \
        --learned_attn \
        --augment
    else
        set -ex
        python train.py  \
        --dataroot $data \
        --name $exp \
        --model sc \
        --gpu_ids 1 \
        --lambda_spatial 10 \
        --lambda_gradient 0 \
        --attn_layers 4,7,9 \
        --loss_mode cos \
        --gan_mode lsgan \
        --display_port 8093 \
        --direction AtoB \
        --patch_size 64
    fi
else
    echo "Not LSeSim model"
fi
