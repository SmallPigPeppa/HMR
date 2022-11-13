import h5py
import numpy
import numpy as np
from kpts_mapping.gta import GTA_ORIGINAL_NAMES, GTA_KEYPOINTS
from kpts_mapping.gta_im import GTA_IM_NPZ_KEYPOINTS, GTA_IM_PKL_KEYPOINTS
from kpts_mapping.smplx import SMPLX_KEYPOINTS
import pickle
import os
import cv2

# step0:
# merge info_frames.pickle info_frames.npz
# trans world to camera


data_root = './'
rec_idx = '2020-06-11-10-06-48'
info_pkl = pickle.load(open(os.path.join(data_root, rec_idx, 'info_frames.pickle'), 'rb'))
info_npz = np.load(open(os.path.join(data_root, rec_idx, 'info_frames.npz'), 'rb'))
kpts_npz = np.array(info_npz['joints_3d_world'])
kpts_pkl = np.array([i['kpvalue'] for i in info_pkl]).reshape(kpts_npz.shape[0], -1, kpts_npz.shape[2])
kpts_pkl_names = [i['kpname'] for i in info_pkl]
world2cam = np.array(info_npz['world2cam_trans'])
kpts_world = np.concatenate((kpts_npz, kpts_pkl), axis=1)
kpts_valid = np.zeros([len(kpts_world), len(kpts_world[0]), 1])
print(
    f'loading kpts...\nkpts_npz.shape, kpts_pkl.shape, kpts_world.shape: {kpts_npz.shape}, {kpts_pkl.shape}, {kpts_world.shape}')
kpts_camera = []
for i in range(len(kpts_world[0])):
    if sum(kpts_world[0, i, :]) != 0:
        kpts_valid[:, i] = 1
for i in range(len(kpts_world)):
    r_i = world2cam[i][:3, :3].T
    t_i = world2cam[i][3, :3]
    # cam_point_i = r_i * kpts_world[i] + t_i
    cam_point_i = [np.matmul(r_i, kpt) + t_i for kpt in kpts_world[i]]
    kpts_camera.append(cam_point_i)
kpts_camera = np.concatenate((kpts_camera, kpts_valid), axis=-1)

# step1
# gta_im to gta
kpts_gta_im = numpy.array(kpts_camera)
kpts_gta = numpy.zeros(shape=(len(kpts_gta_im), len(GTA_KEYPOINTS), 4))
gta_im_names = GTA_IM_NPZ_KEYPOINTS + GTA_IM_PKL_KEYPOINTS
mapping_list = []

for i in range(len(kpts_gta)):
    mapping_list_i = []
    gta_im_names = GTA_IM_NPZ_KEYPOINTS + kpts_pkl_names[i]
    for kpt_name in GTA_ORIGINAL_NAMES:
        if kpt_name not in gta_im_names:
            mapping_list_i.append(-1)
        else:
            mapping_list_i.append(gta_im_names.index(kpt_name))
    mapping_list.append(mapping_list_i)

for i in range(len(kpts_gta)):
    for j in range(len(kpts_gta[0])):
        if mapping_list[i][j] != -1:
            kpts_gta[i][j] = kpts_gta_im[i][mapping_list[i][j]]
# average for nose
for i in range(len(kpts_gta)):
    kpts_gta[i][-1] = np.average(kpts_gta[i][45:51], axis=0)

# step2
# gta to smplx
mapping_list2 = []
valid_len = 0
for kpt_name in SMPLX_KEYPOINTS:
    if kpt_name not in GTA_KEYPOINTS:
        mapping_list2.append(-1)
    else:
        mapping_list2.append(GTA_KEYPOINTS.index(kpt_name))
        valid_len += 1
print(f'tansform to smplx joints, num kpts: {len(mapping_list2)}')
# print(mapping_list2)
kpts_smplx = numpy.zeros(shape=(len(kpts_gta), len(SMPLX_KEYPOINTS), 4))
for i in range(len(kpts_smplx)):
    for j in range(len(kpts_smplx[0])):
        if mapping_list2[j] != -1:
            kpts_smplx[i][j] = kpts_gta[i][mapping_list2[j]]

# step3: save smplx as hdf5
h5f = h5py.File(os.path.join(data_root, rec_idx, 'annot.h5'), 'w')
gt3d = kpts_smplx[:, :23, :]
print(gt3d.shape)
gt2d=[]
rvec = np.zeros([3, 1], dtype=float)
tvec = np.zeros([3, 1], dtype=float)
dist_coeffs = np.zeros([4,1], dtype=float)
for i in range(len(gt3d)):
    camera_matrix_i = info_npz['intrinsics'][i]
    gt3d_i=np.array(gt3d[i,:,:3],dtype=float)
    gt2d_i, _ = cv2.projectPoints(gt3d_i, rvec, tvec, camera_matrix_i, dist_coeffs)
    gt2d.append(gt2d_i)
gt2d=np.squeeze(np.array(gt2d,dtype=float))
gt3d=np.array(gt3d,dtype=float)
valid_kpts=gt3d[:,:,-1].reshape(len(gt3d),-1,1)
print(gt2d.shape,valid_kpts.shape)
gt2d = np.concatenate((gt2d,valid_kpts ), axis=-1)

h5f.create_dataset('gt2d', data=gt2d)
h5f.create_dataset('gt3d', data=gt3d)

# step3: get smplx mesh pose and shape
smplx_parms = pickle.load(open(os.path.join(data_root, rec_idx, '001_all.pkl'), 'rb'))
shape = smplx_parms['betas']  # 1,10
left_hand_pose = smplx_parms['left_hand_pose']  # 955,12
right_hand_pose = smplx_parms['right_hand_pose']  # 955,12
global_orient = smplx_parms['global_orient']  # 955,3
transl = smplx_parms['transl']  # 955,3
jaw_pose = smplx_parms['jaw_pose']  # 955,3
leye_pose = smplx_parms['leye_pose']  # 955,3
reye_pose = smplx_parms['reye_pose']  # 955,3
body_pose = smplx_parms['body_pose']  # 955,63


shape = np.repeat(shape, len(kpts_smplx), axis=0)
h5f.create_dataset('shape', data=shape)
h5f.create_dataset('pose', data=body_pose)
h5f.close()
with h5py.File(os.path.join(data_root, rec_idx, 'annot.h5'), "r+") as f:
    print("Keys: %s" % f.keys())

