import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose
from src.datasets.definitions import *


META_DIMS = 'meta_dims'


class Identity:
    def __call__(self, sample):
        return sample


class AssertModalitiesEqualDimensionsPIL:
    def __call__(self, sample):
        mod_ref, dims = None, None
        for modality, img in sample.items():
            if not isinstance(img, Image.Image):
                continue
            if dims is None:
                mod_ref, dims = modality, img.size
            else:
                assert img.size == dims, \
                    f'Size of {modality} is {img.size} does not match size of {mod_ref} which is {dims}'
        sample[META_DIMS] = dims
        return sample


class CropForPassableSidesPIL:
    def __init__(self, multiple):
        self.multiple = multiple

    def __call__(self, sample):
        assert META_DIMS in sample, 'AssertModalitiesEqualDimensionsPIL is required'
        width, height = sample[META_DIMS]
        width_new = self.multiple * (width // self.multiple)
        height_new = self.multiple * (height // self.multiple)
        assert width_new > 0 and height_new > 0, 'Bad input dimensions'
        for modality in sample.keys():
            data = sample[modality]
            if isinstance(data, Image.Image):
                data = data.crop((0, 0, width_new, height_new))
                sample[modality] = data
        sample[META_DIMS] = width_new, height_new
        return sample


class _RandomScaledTiltedWarpedPIL:
    def __init__(
            self, dst_size, random_scale_min, random_scale_max,
            random_tilt_max_deg, random_wiggle_max_ratio, random_reflect,
            semseg_ignore_class, center_offset_instead_of_random
    ):
        assert isinstance(dst_size, int) or isinstance(dst_size, tuple), f'Invalid type of dst_size {type(dst_size)}'
        self.dst_size = dst_size if isinstance(dst_size, tuple) else (dst_size, dst_size)
        self.random_scale_min = random_scale_min
        self.random_scale_max = random_scale_max
        self.random_tilt_max_deg = random_tilt_max_deg
        self.random_wiggle_max_ratio = random_wiggle_max_ratio
        self.random_reflect = random_reflect
        self.semseg_ignore_class = semseg_ignore_class
        self.center_offset_instead_of_random = center_offset_instead_of_random

    def __call__(self, sample):
        assert MOD_RGB in sample, f'{__class__} needs to know canvas size'
        src_size = sample[MOD_RGB].size
        dst_corners = [
            np.array([0, 0], dtype=np.float32),
            np.array([0, self.dst_size[1]], dtype=np.float32),
            np.array([self.dst_size[0], self.dst_size[1]], dtype=np.float32),
            np.array([self.dst_size[0], 0], dtype=np.float32)
        ]
        if self.random_reflect:
            do_reflect = np.random.random() < 0.5
            if do_reflect:
                dst_corners = list(reversed(dst_corners))
        src_corners, src_scale = _RandomScaledTiltedWarpedPIL._generate_corners(
            src_size, self.dst_size, self.random_scale_min, self.random_scale_max,
            self.random_tilt_max_deg, self.random_wiggle_max_ratio,
            self.center_offset_instead_of_random,
        )
        warp_coef_inv = _RandomScaledTiltedWarpedPIL._perspective_transform_from_corners(dst_corners, src_corners)

        for modality, data in sample.items():
            if modality not in MODE_INTERP:
                continue
            interp = MODE_INTERP[modality]
            if interp is None:
                continue
            elif interp in ('nearest', 'bilinear'):
                assert isinstance(data, Image.Image), f'Input must be PIL.Image, found {type(data)}'
                interp_pil = {
                    'nearest': Image.NEAREST,
                    'bilinear': Image.BILINEAR,
                }[interp]
                fill_color = {
                    MOD_RGB: None,
                    MOD_SS_DENSE: self.semseg_ignore_class,
                }[modality]
                data = data.transform(
                    self.dst_size, Image.PERSPECTIVE, warp_coef_inv, interp_pil, fillcolor=fill_color
                )
            sample[modality] = data

        sample[META_DIMS] = self.dst_size

        return sample

    # adopted from https://stackoverflow.com/questions/14177744/how-does-perspective-transformation-work-in-pil
    # arguments must be lists of tuples like [(x_tl, y_tl), (x_bl, y_bl), (x_br, y_br), (x_tr, y_tr)] with frame of
    # reference in the top-left corner of the image
    @staticmethod
    def _perspective_transform_from_corners(corners_src, corners_dst):
        matrix = []
        for p_src, p_dst in zip(corners_src, corners_dst):
            matrix.append([p_src[0], p_src[1], 1, 0, 0, 0, -p_dst[0] * p_src[0], -p_dst[0] * p_src[1]])
            matrix.append([0, 0, 0, p_src[0], p_src[1], 1, -p_dst[1] * p_src[0], -p_dst[1] * p_src[1]])
        A = np.matrix(matrix, dtype=np.float64)
        B = np.array(corners_dst).reshape(8)
        res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
        return np.array(res, dtype=np.float32).reshape(8)

    @staticmethod
    def _dst_corners_bounding_box(corners):
        x_min, x_max = corners[0][0], corners[0][0]
        y_min, y_max = corners[0][1], corners[0][1]
        for corner in corners[1:]:
            x_min = min(x_min, corner[0])
            x_max = max(x_max, corner[0])
            y_min = min(y_min, corner[1])
            y_max = max(y_max, corner[1])
        return x_min, x_max, y_min, y_max

    @staticmethod
    def _transform_scale_rotate_wiggle(dst_size, scale_min, scale_max, angle_max_deg, wiggle_max_ratio):
        corners = [
            np.array([-dst_size[0] / 2, -dst_size[1] / 2], dtype=np.float32),
            np.array([-dst_size[0] / 2, dst_size[1] / 2], dtype=np.float32),
            np.array([dst_size[0] / 2, dst_size[1] / 2], dtype=np.float32),
            np.array([dst_size[0] / 2, -dst_size[1] / 2], dtype=np.float32)
        ]

        max_wiggle_pix = wiggle_max_ratio * min(dst_size[0], dst_size[1]) / 2
        scale = np.random.uniform(scale_min, scale_max)
        angle_deg = np.random.uniform(-angle_max_deg, angle_max_deg) if 0 < angle_max_deg <= 45 else 0
        wiggle_factor = [
            np.array([
                np.random.uniform(-max_wiggle_pix, max_wiggle_pix),
                np.random.uniform(-max_wiggle_pix, max_wiggle_pix)
            ], dtype=np.float32) for _ in range(4)
        ]

        angle_rad = np.deg2rad(angle_deg)
        matrix_rot = np.array([
            [np.cos(angle_rad), np.sin(-angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)]
        ], dtype=np.float32)

        corners = [np.matmul(matrix_rot, scale * (c + w)) for c, w in zip(corners, wiggle_factor)]
        return corners, scale

    @staticmethod
    def _generate_corners(
            src_size, dst_size, random_scale_min=1.0, random_scale_max=2.0,
            random_tilt_max_deg=0.0, random_wiggle_max_ratio=0.0, center_offset_instead_of_random=False
    ):
        assert random_scale_min > 0, 'random_scale_min must be positive'
        assert random_scale_max >= random_scale_min, 'random_scale_max > random_scale_min'
        assert random_tilt_max_deg >= 0, 'tilt must be non negative'
        assert 0 <= random_wiggle_max_ratio < 0.5, 'random_wiggle_max_ratio must be [0, 1/2]'

        corners, scale = _RandomScaledTiltedWarpedPIL._transform_scale_rotate_wiggle(
            dst_size, random_scale_min, random_scale_max, random_tilt_max_deg, random_wiggle_max_ratio
        )
        x_min, x_max, y_min, y_max = _RandomScaledTiltedWarpedPIL._dst_corners_bounding_box(corners)

        range_x_min = -x_min
        range_x_max = src_size[0] - x_max
        range_y_min = -y_min
        range_y_max = src_size[1] - y_max

        if center_offset_instead_of_random or range_x_max <= range_x_min:
            offs_x = (range_x_min + range_x_max) * 0.5
        else:
            offs_x = np.random.uniform(range_x_min, range_x_max)

        if center_offset_instead_of_random or range_y_max <= range_y_min:
            offs_y = (range_y_min + range_y_max) * 0.5
        else:
            offs_y = np.random.uniform(range_y_min, range_y_max)

        corners = [c + np.array([offs_x, offs_y], dtype=np.float32) for c in corners]
        return corners, scale


class RandomScaledTiltedWarpedCropPIL(_RandomScaledTiltedWarpedPIL):
    def __init__(
            self, dst_size, random_scale_min, random_scale_max, random_tilt_max_deg, random_wiggle_max_ratio,
            random_reflect, semseg_ignore_class
    ):
        super(RandomScaledTiltedWarpedCropPIL, self).__init__(
            dst_size, random_scale_min, random_scale_max, random_tilt_max_deg,
            random_wiggle_max_ratio, random_reflect, semseg_ignore_class,
            center_offset_instead_of_random=False
        )


class CropCenterPIL(_RandomScaledTiltedWarpedPIL):
    def __init__(self, dst_size, semseg_ignore_class):
        super(CropCenterPIL, self).__init__(
            dst_size,
            random_scale_min=1.0,
            random_scale_max=1.0,
            random_tilt_max_deg=0.0,
            random_wiggle_max_ratio=0.0,
            random_reflect=False,
            semseg_ignore_class=semseg_ignore_class,
            center_offset_instead_of_random=True,
        )


class ConvertToTensor:
    def __call__(self, sample):
        for modality in sample.keys():
            data = sample[modality]
            if modality == MOD_ID:
                data = torch.tensor(data, dtype=torch.long)
            elif modality == MOD_RGB:
                data = torch.from_numpy(np.array(data)).float().permute(2, 0, 1)
            elif modality == MOD_SS_DENSE:
                data = torch.from_numpy(np.array(data)).long().unsqueeze(0)
            elif modality == MOD_DEPTH:
                pass  # torch.Tensor beginning from Dataset
            elif modality == META_DIMS:
                pass  # handled in CleanUp
            else:
                print(f'Unaccounted data, collate WILL complain: {modality} {data}')
            sample[modality] = data
        return sample


class ZeroMeanUnitVarianceRgbTensor:
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, sample):
        if MOD_RGB not in sample:
            return sample
        assert torch.is_tensor(sample[MOD_RGB]), f'Invalid RGB modality type {type(sample[MOD_RGB])}'
        mean = torch.tensor(self.mean).view(3, 1, 1)
        stddev = torch.tensor(self.stddev).view(3, 1, 1)
        sample[MOD_RGB] = (sample[MOD_RGB] - mean) / stddev
        return sample


class ZeroMeanUnitVarianceDepthTensor:
    def __init__(self, mean, stddev):
        self.mean = mean
        self.stddev = stddev

    def __call__(self, sample):
        if MOD_DEPTH not in sample:
            return sample
        m = sample[MOD_DEPTH]
        assert type(m) is dict and 'type' in m and 'yx' in m and 'val' in m, f'Invalid DEPTH: {m}'
        sample[MOD_DEPTH]['val'] = (sample[MOD_DEPTH]['val'] - self.mean) / self.stddev
        return sample


class CleanUp:
    def __call__(self, sample):
        del sample[META_DIMS]
        return sample


def get_transforms(
    semseg_ignore_class=None,
    geom_scale_min=0.5,
    geom_scale_max=2.0,
    geom_tilt_max_deg=0,
    geom_wiggle_max_ratio=0,
    geom_reflect=True,
    crop_for_passable=0,
    crop_random=0,
    crop_center=0,
    rgb_zero_mean_status=False,
    rgb_mean=None,
    rgb_stddev=None,
    depth_zero_mean_status=False,
    depth_mean=None,
    depth_stddev=None,
):
    return Compose([
        AssertModalitiesEqualDimensionsPIL(),
        CropForPassableSidesPIL(crop_for_passable) if crop_for_passable > 0 else Identity(),
        RandomScaledTiltedWarpedCropPIL(
            crop_random,
            random_scale_min=geom_scale_min,
            random_scale_max=geom_scale_max,
            random_tilt_max_deg=geom_tilt_max_deg,
            random_wiggle_max_ratio=geom_wiggle_max_ratio,
            random_reflect=geom_reflect,
            semseg_ignore_class=semseg_ignore_class,
        ) if crop_random > 0 else Identity(),
        CropCenterPIL(
            crop_center,
            semseg_ignore_class=semseg_ignore_class,
        ) if crop_center > 0 else Identity(),
        ConvertToTensor(),
        ZeroMeanUnitVarianceRgbTensor(rgb_mean, rgb_stddev) if rgb_zero_mean_status else Identity(),
        ZeroMeanUnitVarianceDepthTensor(depth_mean, depth_stddev) if depth_zero_mean_status else Identity(),
        CleanUp()
    ])
