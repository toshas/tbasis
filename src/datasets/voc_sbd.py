import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import VOCSegmentation, SBDataset

from src.datasets.definitions import *


class DatasetVocSbd(Dataset):
    """
    This dataset extends Pascal VOC [1] with Semantic Boundaries Dataset [2]. It provides 10582 training and 1449
    validation samples. The training split is a union of VOC2012 training split and all of SBD images,
    excluding VOC2012 validation images. The validation split is identical to VOC2012 validation split.
    [1]: @misc{everingham2012pascal,
             author={Everingham, M. and {Van Gool}, L. and Williams, C. K. I. and Winn, J. and Zisserman, A.},
             title={The {PASCAL} {V}isual {O}bject {C}lasses {C}hallenge 2012 {(VOC2012)} {R}esults},
             howpublished="http://www.pascal-network.org/challenges/VOC/voc2012/workshop/index.html"
             year={2012}
         }
    [2]: @inproceedings{bharath2011semanticcontours,
             author={Bharath Hariharan and Pablo Arbelaez and Lubomir Bourdev and Subhransu Maji and Jitendra Malik},
             title={Semantic Contours from Inverse Detectors},
             booktitle={ICCV},
             year={2011},
         }
    """

    def __init__(self, dataset_root, split, download=True):
        assert split in (SPLIT_TRAIN, SPLIT_VALID), f'Invalid split {split}'

        root_voc = os.path.join(dataset_root, 'VOC')
        root_sbd = os.path.join(dataset_root, 'SBD')

        self.ds_voc_valid = VOCSegmentation(root_voc, image_set=SPLIT_VALID, download=download)

        if split == SPLIT_TRAIN:
            self.ds_voc_train = VOCSegmentation(root_voc, image_set=SPLIT_TRAIN, download=False)
            self.ds_sbd_train = SBDataset(
                root_sbd,
                image_set=SPLIT_TRAIN,
                download=download and not os.path.isdir(os.path.join(root_sbd, 'img'))
            )
            self.ds_sbd_valid = SBDataset(root_sbd, image_set=SPLIT_VALID, download=False)

            self.name_to_ds_id = {
                self._sample_name(path): (self.ds_sbd_train, i) for i, path in enumerate(self.ds_sbd_train.images)
            }
            self.name_to_ds_id.update({
                self._sample_name(path): (self.ds_sbd_valid, i) for i, path in enumerate(self.ds_sbd_valid.images)
            })
            self.name_to_ds_id.update({
                self._sample_name(path): (self.ds_voc_train, i) for i, path in enumerate(self.ds_voc_train.images)
            })
            for path in self.ds_voc_valid.images:
                name = self._sample_name(path)
                self.name_to_ds_id.pop(name, None)
        else:
            self.name_to_ds_id = {
                self._sample_name(path): (self.ds_voc_valid, i) for i, path in enumerate(self.ds_voc_valid.images)
            }

        self.sample_names = list(sorted(self.name_to_ds_id.keys()))
        self.sample_id_to_name = {k: v for k, v in enumerate(sorted(self.name_to_ds_id.keys()))}
        self.sample_name_to_id = {v: k for k, v in self.sample_id_to_name.items()}
        self.transforms = None

    def set_transforms(self, transforms):
        self.transforms = transforms

    def get(self, index, override_transforms=None):
        ds, idx = self.name_to_ds_id[self.name_from_index(index)]

        path_rgb = ds.images[idx]
        name = self._sample_name(path_rgb)

        rgb = Image.open(path_rgb).convert('RGB')

        ss_dense_path = ds.masks[idx]
        if ss_dense_path.endswith('mat'):
            ss_dense = ds._get_segmentation_target(ss_dense_path)
        else:
            ss_dense = Image.open(ss_dense_path)
        assert ss_dense.size == rgb.size, f'RGB and SEMSEG shapes do not match in sample {name}'

        out = {
            MOD_ID: index,
            MOD_RGB: rgb,
            MOD_SS_DENSE: ss_dense,
        }

        if override_transforms is not None:
            out = override_transforms(out)
        elif self.transforms is not None:
            out = self.transforms(out)

        return out

    def name_from_index(self, index):
        return self.sample_names[index]

    def __getitem__(self, index):
        return self.get(index)

    def __len__(self):
        return len(self.sample_names)

    @staticmethod
    def _sample_name(path):
        return path.split('/')[-1].split('.')[0]

    def index2name(self, index):
        return self.sample_id_to_name[index]

    def name2index(self, name):
        return self.sample_name_to_id[name]

    @property
    def num_classes(self):
        return 21

    @property
    def ignore_label(self):
        return 255

    @property
    def rgb_mean(self):
        return [255 * 0.485, 255 * 0.456, 255 * 0.406]

    @property
    def rgb_stddev(self):
        return [255 * 0.229, 255 * 0.224, 255 * 0.225]

    @property
    def semseg_class_colors(self):
        return [
            (0, 0, 0),  # 'background'
            (128, 0, 0),  # 'plane'
            (0, 128, 0),  # 'bike'
            (128, 128, 0),  # 'bird'
            (0, 0, 128),  # 'boat'
            (128, 0, 128),  # 'bottle'
            (0, 128, 128),  # 'bus'
            (128, 128, 128),  # 'car'
            (64, 0, 0),  # 'cat'
            (192, 0, 0),  # 'chair'
            (64, 128, 0),  # 'cow'
            (192, 128, 0),  # 'table'
            (64, 0, 128),  # 'dog'
            (192, 0, 128),  # 'horse'
            (64, 128, 128),  # 'motorbike'
            (192, 128, 128),  # 'person'
            (0, 64, 0),  # 'plant'
            (128, 64, 0),  # 'sheep'
            (0, 192, 0),  # 'sofa'
            (128, 192, 0),  # 'train'
            (0, 64, 128),  # 'monitor'
        ]

    @property
    def semseg_class_names(self):
        return [
            'background',
            'plane',
            'bike',
            'bird',
            'boat',
            'bottle',
            'bus',
            'car',
            'cat',
            'chair',
            'cow',
            'table',
            'dog',
            'horse',
            'motorbike',
            'person',
            'plant',
            'sheep',
            'sofa',
            'train',
            'monitor',
        ]


if __name__ == '__main__':
    import tempfile
    tmpdir = os.path.join(tempfile.gettempdir(), 'dataset_VOC_SBD')
    print(f'Temporary directory: {tmpdir}')
    print('Checking dataset integrity...')
    voc2012weak_train = DatasetVocSbd(tmpdir, SPLIT_TRAIN)
    voc2012weak_valid = DatasetVocSbd(tmpdir, SPLIT_VALID)
    print('Dataset integrity check passed')
