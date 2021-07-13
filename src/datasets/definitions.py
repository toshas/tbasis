MOD_ID = 'id'
MOD_RGB = 'rgb'
MOD_SS_DENSE = 'semseg_dense'
MOD_DEPTH = 'depth'
MOD_VALIDITY = 'validity_mask'

SPLIT_TRAIN = 'train'
SPLIT_VALID = 'val'

MODE_INTERP = {
    MOD_ID: None,
    MOD_RGB: 'bilinear',
    MOD_SS_DENSE: 'nearest',
    MOD_DEPTH: 'sparse',
    MOD_VALIDITY: 'nearest',
}
