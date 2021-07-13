# Configuring Experiments

## Creating project environment config

The training script allows to specify multiple `yml` config files, which will be concatenated during execution. 
This is done to separate experiment configs from environment configs. 
To start running experiments, create your own config file with a number of environment settings, similar to those 
`configs/env_*.yml`. 
The settings are as follows:

```yaml
```yaml
root_datasets:
  # str: Path where CIFAR-10 will be downloaded upon the first run
  cifar10: /cluster/scratch/obukhova/torchvision
  # str: Path where CIFAR-100 will be downloaded upon the first run
  cifar100: /cluster/scratch/obukhova/torchvision
  # str: Path where MNIST will be downloaded upon the first run
  mnist: /cluster/scratch/obukhova/torchvision
  # str: Path where Pascal VOC and Semantic Boundaries Dataset will be downloaded 
  # upon the first run
  voc_sbd: /cluster/scratch/obukhova/VOC_SBD
# str: Path for local logs, each stored a separate folder named after the config file
root_logs: /raid/obukhova/logs
# str: Path where pretrained tbasis model will be looked for
root_models: /cluster/scratch/obukhova/models
# str: Path for local W&B logs
root_wandb: /raid/obukhova/logs_wandb
# Boolean: Whether datasets should be downloaded before using (downloads only once,
# but may annoy with unpacking)
dataset_download: True
# List: Environment variables to make sure are set
assert_env_set:
    - CUDA_VISIBLE_DEVICES
# int: Interval for loss logging to tensorboard and W&B
num_log_loss_steps: 10
# int: Number of worker threads working in parallel to load training samples
workers: 4
# int: Number of worker threads working in parallel to load validation samples
workers_validation: 4
# str: Name of the project in W&B
wandb_project: tbasis
```

## Classification Configuration Template

The following config template should be used with `train_imgcls.py` training script:

```yaml
# imgcls: Experiment type
experiment: imgcls
# str: __FILENAMEBASE__ will be substituted with the file name, any other 
# string will be used as is
experiment_name: __FILENAMEBASE__
# str: __FILENAMEBASE__ will be substituted with the log dir name, any other 
# string will be used as is (root_logs will be prepended)
log_dir: __FILENAMEBASE__
# mnist | cifar10 | cifar100: Input dataset
dataset: cifar10
# int: Total number of training epochs
num_epochs_train_total: 1000
# int: Training batch size
batch_size: 128
# int: Validation batch size
batch_size_validation: 256

# lenet5 | resnet_cifar10: Base CNN template
model_name: resnet_cifar10
# str: Model specification
model_name_specific: resnet32
# Dict: Extra kwargs for the base model
model_name_specific_kwargs: {}

# Boolean: Whether to perform T-Basis compression
compressed: True
# opt_einsum | hierarchical: Meta program for tensor contraction
contraction_method: opt_einsum
# exp | linear: Activation function for rank adapter values
rank_adaptation: exp
# Boolean: Whether to use TT-Matrix format with Z-order of i/o factors
permute_and_group_factors: True
# List: Names of torch.nn.Module names (relative to the model root) not subject 
# to reparameterization
module_names_ignored:
    model.conv1

# int: Number of basis vectors in T-Basis
basis_size: 8
# int: Rank of TR-cores
basis_rank: 4
# int: Mode of TR-cores
basis_mode: 9
# Boolean: Whether to learn T-Basis parameters
basis_trainable_sts: True
# int: RNG reproducibility seed
basis_init_seed: 2020
# str | Null: A path to load pre-trained T-Basis
basis_pretrained_path: Null

# float: L2 regularization weight on decompressed weight matrices elements
decompressed_L2_regularizer_coef: 3e-4

# adam | sgd: Type of model optimizer
optimizer_model_name: sgd
# Dict: Extra kwargs for model optimizer (must at least have lr)
optimizer_model_kwargs:
    lr: 0.1
    momentum: 0.9
    weight_decay: 0.0001

# adam | sgd: Type of basis and coefficients optimizer
optimizer_basis_name: adam
# Dict: Extra kwargs for model and coefficients optimizer (must at least have lr)
optimizer_basis_kwargs:
    lr: 0.01
    betas: [0.95, 0.9995]

# List: MultiStep LR scheduler milestones as fractions of training duration (example below: 50%, 75%)
lr_scheduler_milestones:
    - 0.5
    - 0.75
# float: MultiStep LR scheduler decay factor
lr_scheduler_gamma: 0.1
# Boolean: Whether to perform LR warmup (gradually increase LR from 0 to init)
lr_warmup_sts: True
# int: Number of warmup steps
lr_warmup_steps: 2000
```

## Semantic Segmentation Configuration Template

The following config template should be used with `train_semseg.py` training script:

```yaml
# semseg: Experiment type
experiment: semseg
# str: __FILENAMEBASE__ will be substituted with the file name, any other 
# string will be used as is
experiment_name: __FILENAMEBASE__
# str: __FILENAMEBASE__ will be substituted with the log dir name, any other 
# string will be used as is (root_logs will be prepended)
log_dir: __FILENAMEBASE__
# voc_sbd: Pascal VOC + Semantic Boundaries
dataset: voc_sbd
# int: Total number of training steps
num_training_steps: 360000
# int: Interval between validations
num_validation_steps: 5000
# int: Training batch size
batch_size: 16
# int: Validation batch size
batch_size_validation: 32

# deeplabv3p: Deeplab V3+
model_name: deeplabv3p
# str: Model encoder base network name
model_encoder_name: resnet34
# Boolean: Model encoder pretrained weights flag
model_encoder_pretrained: False

# int: Augmentation crop size
aug_input_crop_size: 384
# float: Augmentation min scale range
aug_geom_scale_min: 0.5
# float: Augmentation max scale range
aug_geom_scale_max: 2.0
# float: Augmentation max tilt angle
aug_geom_tilt_max_deg: 0
# float: Augmentation max bounding box corner displacement as a ratio of crop size
aug_geom_wiggle_max_ratio: 0
# Boolean: Augmentation reflection flag
aug_geom_reflect: True
# Boolean: Validation augmentation center cropping flag 
aug_geom_validation_center_crop_sts: True
# Boolean: Validation augmentation center cropping size
aug_geom_validation_center_crop_size: 512

# Boolean: Whether to perform T-Basis compression
compressed: True
# opt_einsum | hierarchical: Meta program for tensor contraction
contraction_method: hierarchical
# exp | linear: Activation function for rank adapter values
rank_adaptation: exp
# Boolean: Whether to use TT-Matrix format with Z-order of i/o factors
permute_and_group_factors: True
# List: Names of torch.nn.Module names (relative to the model root) not subject 
# to reparameterization
module_names_ignored:
    model.conv1

# int: Number of basis vectors in T-Basis
basis_size: 128
# int: Rank of TR-cores
basis_rank: 16
# int: Mode of TR-cores
basis_mode: 9
# Boolean: Whether to learn T-Basis parameters
basis_trainable_sts: True
# int: RNG reproducibility seed
basis_init_seed: 2020
# str | Null: A path to load pre-trained T-Basis
basis_pretrained_path: Null

# float: L2 regularization weight on decompressed weight matrices elements
decompressed_L2_regularizer_coef: 3e-4

# adam | sgd: Type of model optimizer
optimizer_model_name: sgd
# Dict: Extra kwargs for model optimizer (must at least have lr)
optimizer_model_kwargs:
    lr: 0.007
    momentum: 0.9
    weight_decay: 0.0001

# adam | sgd: Type of basis and coefficients optimizer
optimizer_basis_name: sgd
# Dict: Extra kwargs for model and coefficients optimizer (must at least have lr)
optimizer_basis_kwargs:
    lr: 0.007
    momentum: 0.9
    weight_decay: 0.0001

# float: PolyLR power
lr_scheduler_power: 0.9
# Boolean: Whether to perform LR warmup (gradually increase LR from 0 to init)
lr_warmup_sts: True
# int: Number of warmup steps
lr_warmup_steps: 2000

# int: Number of images to visualize from each batch
visualize_num_samples_in_batch: 6
# List: Training ids to visualize upon every validation
observe_train_ids:
    - 2100
    - 699
    - 5480
# List: Validation ids to visualize upon every validation
observe_valid_ids:
    - 0
    - 100
    - 734
    - 718
    - 389
# int: Width of visualization in tiled images
tensorboard_img_grid_width: 16
```
