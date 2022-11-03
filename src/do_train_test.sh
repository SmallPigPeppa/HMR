

#CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainer.py

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 trainer.py \
--smpl-mean-theta-path /share/wenzhuoliu/code/HMR/model/neutral_smpl_mean_params.h5 \
--smpl-model /share/wenzhuoliu/code/HMR/model/neutral_smpl_with_cocoplus_reg.pkl \
--save-folder /share/wenzhuoliu/code/HMR/out-model \
