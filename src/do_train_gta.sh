

#CUDA_VISIBLE_DEVICES=4,5,6,7 python3 trainer.py

CUDA_VISIBLE_DEVICES=0 python trainer.py \
--smpl-mean-theta-path HMR-data/neutral_smpl_mean_params.h5 \
--smpl-model HMR-data/neutral_smpl_with_cocoplus_reg.pkl \
--save-folder HMR-data/out-model \

sleep 10000
