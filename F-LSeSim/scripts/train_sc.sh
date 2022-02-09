set -ex
python train.py  \
--dataroot ./../data/example/ \
--name example \
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
#\--learned_attn \--augment