import os
import os.path
import json
import cv2
import numpy as np
import torch
import pickle
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
from imgaug import augmenters as iaa

from lib.configs.config import cfg


class GTADataset(Dataset):
    def __init__(self, opt, dataset_name=None):
        super(GTADataset, self).__init__()
        self.opt = opt
        self.root = opt.dataroot
        self.dataset_name = dataset_name
        self.rgb_paths, self.depth_paths, self.mask_paths, self.cam_near_clips, self.cam_far_clips, self.info_pkl, _ = self.getData()
        print(self.info_npz.files)
        self.data_size = len(self.info_pkl)
        self.curriculum_list = list(np.random.choice(self.data_size, self.data_size, replace=False))
        # print("data_size", self.data_size)

    def getData(self):
        data_path = os.path.join(self.root, self.dataset_name)
        self.data_path = data_path
        depth_paths = []
        rgb_paths = []
        mask_paths = []
        # keypoints = []
        cam_near_clips = []
        cam_far_clips = []
        self.info_pkl = pickle.load(open(os.path.join(data_path, 'info_frames.pickle'), 'rb'))
        self.info_npz = np.load(os.path.join(data_path, 'info_frames.npz'),allow_pickle=True)
        # joints_2d = np.load(os.path.join(data_path, 'info_frames.npz'))['joints_2d']
        info_pkl = pickle.load(open(os.path.join(data_path, 'info_frames.pickle'), 'rb'))
        info_npz = np.load(os.path.join(data_path, 'info_frames.npz'))
        for idx in range(0, len(info_pkl)):
            if os.path.exists(os.path.join(data_path, '{:05d}'.format(idx) + '.jpg')):
                # keypoint = joints_2d[idx]
                rgb_path = os.path.join(data_path, '{:05d}'.format(idx) + '.jpg')
                depth_path = os.path.join(data_path, '{:05d}'.format(idx) + '.png')
                mask_path = os.path.join(data_path, '{:05d}'.format(idx) + '_id.png')

                infot = info_pkl[idx]
                cam_near_clip = infot['cam_near_clip']
                if 'cam_far_clip' in infot.keys():
                    cam_far_clip = infot['cam_far_clip']
                else:
                    cam_far_clip = 800.

                # keypoints.append(keypoint)
                rgb_paths.append(rgb_path)
                depth_paths.append(depth_path)
                mask_paths.append(mask_path)
                cam_near_clips.append(cam_near_clip)
                cam_far_clips.append(cam_far_clip)

        return rgb_paths, depth_paths, mask_paths, cam_near_clips, cam_far_clips, info_pkl, info_npz

    def __getitem__(self, anno_index):
        if 'train' in self.opt.phase:
            data = self.online_aug(anno_index)
        else:
            data = self.load_test_data(anno_index)
        return data

    def read_depthmap(self, name, cam_near_clip, cam_far_clip):
        depth = cv2.imread(name)
        depth = np.concatenate(
            (depth, np.zeros_like(depth[:, :, 0:1], dtype=np.uint8)), axis=2
        )
        depth.dtype = np.uint32
        depth = 0.05 * 1000 / depth.astype('float')
        depth = (
                cam_near_clip
                * cam_far_clip
                / (cam_near_clip + depth * (cam_far_clip - cam_near_clip))
        )
        return np.squeeze(depth)

    def load_test_data(self, anno_index):
        """
        Augment data for training online randomly. The invalid parts in the depth map are set to -1.0, while the parts
        in depth bins are set to cfg.MODEL.DECODER_OUTPUT_C + 1.
        :param anno_index: data index.
        """
        rgb_path = self.rgb_paths[anno_index]
        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # bgr, H*W*C
        depth = self.read_depthmap(name=self.depth_paths[anno_index], cam_near_clip=self.cam_near_clips[anno_index],
                                   cam_far_clip=self.cam_far_clips[anno_index])
        drange = depth.max()
        depth_norm = depth / drange
        mask_valid = (depth_norm > 1e-8).astype(np.float)

        rgb_resize = cv2.resize(rgb, (cfg.DATASET.CROP_SIZE[1], cfg.DATASET.CROP_SIZE[0]),
                                interpolation=cv2.INTER_LINEAR)
        # to torch, normalize
        rgb_torch = self.scale_torch(rgb_resize.copy())
        # normalize disp and depth
        depth_normal = depth_norm / (depth_norm.max() + 1e-8)
        depth_normal[~mask_valid.astype(np.bool)] = 0

        data = {'rgb': rgb_torch, 'gt_depth': depth_normal}
        return data

    def online_aug(self, anno_index):
        """
        Augment data for training online randomly.
        :param anno_index: data index.
        """
        # print(anno_index)
        rgb_path = self.rgb_paths[anno_index]
        depth_path = self.depth_paths[anno_index]
        rgb = cv2.imread(rgb_path)[:, :, ::-1]  # rgb, H*W*C
        # joints_2d = np.load(os.path.join(self.data_path, 'info_frames.npz'))['joints_2d'][anno_index]
        # joints_3d_cam = np.load(os.path.join(self.data_path, 'info_frames.npz'))['joints_3d_cam'][anno_index]
        # joints_3d_world = np.load(os.path.join(self.data_path, 'info_frames.npz'))['joints_3d_world'][anno_index]
        # world2cam_trans = np.load(os.path.join(self.data_path, 'info_frames.npz'))['world2cam_trans'][anno_index]
        # intrinsics = np.load(os.path.join(self.data_path, 'info_frames.npz'))['intrinsics'][anno_index]
        # print("#############################debug############################")
        # print(self.info_npz)
        # print("#############################debug############################")
        # joints_2d = self.info_npz['joints_2d'][anno_index]
        # joints_3d_cam = self.info_npz['joints_3d_cam'][anno_index]
        # joints_3d_world = self.info_npz['joints_3d_world'][anno_index]
        # world2cam_trans = self.info_npz['world2cam_trans'][anno_index]
        # intrinsics = self.info_npz['intrinsics'][anno_index]
        joints_2d = 0
        joints_3d_cam = 0
        joints_3d_world = 0
        world2cam_trans = 0
        intrinsics = 0
        # focal_length = (intrinsics[0][0]).astype(np.float32)
        focal_length = (np.array(10.)).astype(np.float32)
        depth, invalid_depth, sem_mask = self.load_training_data(anno_index)

        rgb_aug = self.rgb_aug(rgb)

        # resize rgb, depth, disp
        flip_flg, resize_size, crop_size, pad, resize_ratio = self.set_flip_resize_crop_pad(rgb_aug)

        rgb_resize = self.flip_reshape_crop_pad(rgb_aug, flip_flg, resize_size, crop_size, pad, 0)
        # depth_resize = self.flip_reshape_crop_pad(depth, flip_flg, resize_size, crop_size, pad, 0,
        #                                           resize_method='nearest')
        depth_resize = self.flip_reshape_crop_pad(depth, flip_flg, resize_size, crop_size, pad, -1,
                                                  resize_method='nearest')
        sem_mask_resize = self.flip_reshape_crop_pad(sem_mask.astype(np.uint8), flip_flg, resize_size, crop_size, pad,
                                                     0, resize_method='nearest')

        # resize sky_mask, and invalid_regions
        invalid_depth_resize = self.flip_reshape_crop_pad(invalid_depth.astype(np.uint8), flip_flg, resize_size,
                                                          crop_size, pad, 0, resize_method='nearest')
        # # resize ins planes
        road_mask = sem_mask == -1
        ins_planes_mask = sem_mask == -1
        ins_planes_mask[road_mask] = int(np.unique(ins_planes_mask).max() + 1)
        ins_planes_mask_resize = self.flip_reshape_crop_pad(ins_planes_mask.astype(np.uint8), flip_flg, resize_size,
                                                            crop_size, pad, 0, resize_method='nearest')
        sky_mask_resize = sem_mask_resize == -1
        human_mask_resize = sem_mask_resize == 126

        # normalize disp and depth
        print(depth_resize.max())
        depth_resize = (depth_resize / (depth_resize.max() + 1e-8)) * 10
        depth_resize = depth_resize + 1e-8

        # invalid regions are set to -1, sky regions are set to 0 in disp and 10 in depth
        # depth_resize[invalid_depth_resize.astype(np.bool) | (depth_resize > 1e7) | (depth_resize < 0)] = -1
        depth_resize[invalid_depth_resize.astype(np.bool) | (depth_resize > 1e7) | (depth_resize < 0)] = 1e-8
        depth_resize[sky_mask_resize.astype(np.bool)] = 20

        # to torch, normalize
        rgb_torch = self.scale_torch(rgb_resize.copy())
        depth_torch = self.scale_torch(depth_resize)
        ins_planes = torch.from_numpy(ins_planes_mask_resize)

        # TODO: add transforms for joints and camera_trans

        data = {
            'rgb': rgb_torch, 'depth': depth_torch, 'sem_mask': torch.tensor(sem_mask_resize),
            'human_mask': torch.tensor(human_mask_resize), 'joints_2d': torch.tensor(joints_2d),
            'joints_3d_cam': torch.tensor(joints_3d_cam), 'joints_3d_world': torch.tensor(joints_3d_world),
            'world2cam_trans': torch.tensor(world2cam_trans), 'intrinsics': torch.tensor(intrinsics),
            'focal_length': torch.tensor(focal_length), 'A_paths': rgb_path, 'B_paths': depth_path,
        }
        return data

    def rgb_aug(self, rgb):
        # data augmentation for rgb
        img_aug = transforms.ColorJitter(brightness=0.0, contrast=0.3, saturation=0.1, hue=0)(Image.fromarray(rgb))
        rgb_aug_gray_compress = iaa.Sequential([iaa.MultiplyAndAddToBrightness(mul=(0.6, 1.25), add=(-20, 20)),
                                                iaa.Grayscale(alpha=(0.0, 1.0)),
                                                iaa.JpegCompression(compression=(0, 70)),
                                                ], random_order=True)
        rgb_aug_blur1 = iaa.AverageBlur(k=((0, 5), (0, 6)))
        rgb_aug_blur2 = iaa.MotionBlur(k=9, angle=[-45, 45])
        img_aug = rgb_aug_gray_compress(image=np.array(img_aug))
        blur_flg = np.random.uniform(0.0, 1.0)
        img_aug = rgb_aug_blur1(image=img_aug) if blur_flg > 0.7 else img_aug
        img_aug = rgb_aug_blur2(image=img_aug) if blur_flg < 0.3 else img_aug
        rgb_colorjitter = np.array(img_aug)
        return rgb_colorjitter

    def set_flip_resize_crop_pad(self, A):
        """
        Set flip, padding, reshaping and cropping flags.
        :param A: Input image, [H, W, C]
        :return: Data augamentation parameters
        """
        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.opt.phase else False

        # reshape
        ratio_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  #
        if 'train' in self.opt.phase:
            resize_ratio = ratio_list[np.random.randint(len(ratio_list))]
        else:
            resize_ratio = 0.5

        resize_size = [int(A.shape[0] * resize_ratio + 0.5),
                       int(A.shape[1] * resize_ratio + 0.5)]  # [height, width]
        # crop
        start_y = 0 if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else np.random.randint(0, resize_size[0] -
                                                                                         cfg.DATASET.CROP_SIZE[0])
        start_x = 0 if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else np.random.randint(0, resize_size[1] -
                                                                                         cfg.DATASET.CROP_SIZE[1])
        crop_height = resize_size[0] if resize_size[0] <= cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0]
        crop_width = resize_size[1] if resize_size[1] <= cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1]
        crop_size = [start_x, start_y, crop_width, crop_height] if 'train' in self.opt.phase else [0, 0, resize_size[1],
                                                                                                   resize_size[0]]

        # pad
        pad_height = 0 if resize_size[0] > cfg.DATASET.CROP_SIZE[0] else cfg.DATASET.CROP_SIZE[0] - resize_size[0]
        pad_width = 0 if resize_size[1] > cfg.DATASET.CROP_SIZE[1] else cfg.DATASET.CROP_SIZE[1] - resize_size[1]
        # [up, down, left, right]
        pad = [pad_height, 0, pad_width, 0] if 'train' in self.opt.phase else [0, 0, 0, 0]
        return flip_flg, resize_size, crop_size, pad, resize_ratio

    def flip_reshape_crop_pad(self, img, flip, resize_size, crop_size, pad, pad_value=0, resize_method='bilinear'):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        # Flip
        if flip:
            img = np.flip(img, axis=1)

        # Resize the raw image
        if resize_method == 'bilinear':
            img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_LINEAR)
        elif resize_method == 'nearest':
            img_resize = cv2.resize(img, (resize_size[1], resize_size[0]), interpolation=cv2.INTER_NEAREST)
        else:
            raise ValueError

        # Crop the resized image
        img_crop = img_resize[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]

        # Pad the raw image
        if len(img.shape) == 3:
            img_pad = np.pad(img_crop, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                             constant_values=(pad_value, pad_value))
        else:
            img_pad = np.pad(img_crop, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                             constant_values=(pad_value, pad_value))
        return img_pad

    def depth_to_bins(self, depth):
        """
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        invalid_mask = depth < 1e-8
        depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX
        bins = ((torch.log10(depth) - cfg.DATASET.DEPTH_MIN_LOG) / cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        bins[bins == cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        depth[invalid_mask] = -1.0
        return bins

    def scale_torch(self, img):
        """
        Scale the image and output it in torch.tensor.
        :param img: input rgb is in shape [H, W, C], input depth/disp is in shape [H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W]
        """
        if len(img.shape) == 2:
            img = img[np.newaxis, :, :]
        if img.shape[2] == 3:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS,
                                                                 cfg.DATASET.RGB_PIXEL_VARS)])
            img = transform(img)
        else:
            img = img.astype(np.float32)
            img = torch.from_numpy(img)
        return img

    def load_training_data(self, anno_index):
        depth = self.read_depthmap(name=self.depth_paths[anno_index], cam_near_clip=self.cam_near_clips[anno_index],
                                   cam_far_clip=self.cam_far_clips[anno_index]).astype(np.uint16)
        # load semantic mask, such as road, sky
        sem_mask = cv2.imread(self.mask_paths[anno_index], cv2.IMREAD_ANYDEPTH).astype(np.uint8)
        sem_mask = np.squeeze(sem_mask)
        invalid_depth = depth < 1e-8

        return depth, invalid_depth, sem_mask

    def loading_check(self, depth, depth_path):
        if 'taskonomy' in depth_path:
            # invalid regions in taskonomy are set to 65535 originally
            depth[depth >= 28000] = 0
        if '3d-ken-burns' in depth_path:
            # maybe sky regions
            depth[depth >= 47000] = 0
        return depth

    def __len__(self):
        return self.data_size

    # def name(self):
    #     return 'DiverseDepth'
