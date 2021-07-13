import math

import numpy as np
import torch
import torch.nn.functional as F


def prime_factors(num):
    factors = []
    for i in range(2, int(np.sqrt(num)) + 1):
        while num % i == 0:
            factors.append(i)
            num //= i
    if num > 1:
        factors.append(num)
    return factors


def explode_qtt_shape(shape, is_batched):
    shape_qtt = [shape[0]] if is_batched else []
    for mode in shape[(1 if is_batched else 0):]:
        shape_qtt += prime_factors(int(mode))
    return tuple(shape_qtt)


def compute_z_order(shape_src, group_tuples=True, pad_groups_at_high_scales=True):
    num_dims_src = len(shape_src)

    factors_src = [prime_factors(dim) for dim in shape_src]
    num_groups = max([len(f) for f in factors_src])

    shape_factors_src = []
    factors_src_id = []

    cur_id = 0
    for i in range(num_dims_src):
        f_len = len(factors_src[i])
        shape_factors_src += factors_src[i]

        cur_factors_src_id = list(range(cur_id, cur_id + f_len))
        cur_id += f_len

        # pad factors_src with dummy dimensions
        if f_len < num_groups:
            dummy_count = (num_groups - f_len)
            if pad_groups_at_high_scales:
                factors_src[i] = [1] * dummy_count + factors_src[i]
                cur_factors_src_id = [-1] * dummy_count + cur_factors_src_id
            else:
                factors_src[i] += [1] * dummy_count
                cur_factors_src_id += [-1] * dummy_count

        factors_src_id.append(cur_factors_src_id)

    factors_src = torch.tensor(factors_src)
    factors_src_id = torch.tensor(factors_src_id)

    factors_dst = factors_src.T
    factors_dst_with_unary_modes = factors_dst
    factors_dst_id = factors_src_id.T

    permute_factors_src_to_dst = factors_dst_id.reshape(-1, ).tolist()
    permute_factors_src_to_dst = [a for a in permute_factors_src_to_dst if a >= 0]

    shape_factors_dst = [a for a in factors_dst.reshape(-1, ).tolist() if a > 1]
    if group_tuples:
        factors_dst = factors_dst.prod(dim=1)
    else:
        factors_dst = factors_dst.reshape(-1)
        factors_dst = factors_dst[factors_dst > 1]
    shape_dst = factors_dst.tolist()
    factors_reverse_id = (torch.ones_like(factors_dst_id) * (factors_dst_id >= 0).int()).reshape(-1, ).cumsum(
        dim=0).view(factors_dst_id.shape)
    factors_reverse_id -= 1
    factors_reverse_id[factors_dst_id < 0] = -1
    factors_reverse_id = factors_reverse_id.T.reshape(-1, ).tolist()
    permute_factors_dst_to_src = [a for a in factors_reverse_id if a >= 0]

    return {
        'shape_src': shape_src,
        'shape_factors_src': shape_factors_src,
        'permute_factors_src_to_dst': permute_factors_src_to_dst,
        'permute_factors_dst_to_src': permute_factors_dst_to_src,
        'shape_factors_dst': shape_factors_dst,
        'shape_dst': shape_dst,
        'factors_dst_with_unary_modes': factors_dst_with_unary_modes,
    }


def explode_spatial_dimensions_to_qtt_z_order(data, num_spatial_dims):
    """
    Performs dimensions factorization and Z-ordering
    Spatial dimensions are expected last
    """
    assert num_spatial_dims > 1, 'Z-order should be used with 2- and more-D data'
    assert torch.is_tensor(data)
    assert data.dim() >= num_spatial_dims

    shape_non_spatial = list(data.shape[:-num_spatial_dims])
    shape_spatial_src = list(data.shape[-num_spatial_dims:])

    shape_plan = compute_z_order(shape_spatial_src)

    shape_factors_src = shape_non_spatial + shape_plan['shape_factors_src']
    shape_dst = shape_non_spatial + shape_plan['shape_dst']

    num_non_spatial = len(shape_non_spatial)
    ids_non_spatial = list(range(num_non_spatial))
    permute_factors_src_to_dst = \
        ids_non_spatial + (torch.tensor(shape_plan['permute_factors_src_to_dst']) + num_non_spatial).tolist()

    data = data.view(shape_factors_src)
    data = data.permute(permute_factors_src_to_dst)
    data = data.reshape(shape_dst)

    return data


def assemble_spatial_dimensions_from_qtt_z_order(data, num_spatial_dims, data_original_shape):
    """
    Performs inverse dimensions factorization and Z-ordering
    Spatial dimensions are expected last
    """
    assert num_spatial_dims > 1, 'Z-order should be used with 2- and more-D data'
    assert torch.is_tensor(data)
    assert data.numel() == torch.tensor(data_original_shape).prod()

    shape_non_spatial = list(data_original_shape[:-num_spatial_dims])

    shape_plan = compute_z_order(data_original_shape[-num_spatial_dims:])

    shape_factors_dst = shape_non_spatial + shape_plan['shape_factors_dst']

    num_non_spatial = len(shape_non_spatial)
    ids_non_spatial = list(range(num_non_spatial))
    permute_factors_dst_to_src = \
        ids_non_spatial + (torch.tensor(shape_plan['permute_factors_dst_to_src']) + num_non_spatial).tolist()

    data = data.view(shape_factors_dst)
    data = data.permute(permute_factors_dst_to_src)
    data = data.reshape(data_original_shape)

    return data


def compute_pad_to_powers_plan(x_shape, base=2):
    if base == 2:
        x_shape_padded = [2 ** int(math.ceil(math.log2(m))) for m in x_shape]
    else:
        x_shape_padded = [base ** int(math.ceil(math.log(m) / math.log(base))) for m in x_shape]
    paddings = [mp - m for mp, m in zip(x_shape_padded, x_shape)]
    return {
        'shape_src': x_shape,
        'paddings': paddings,
        'shape_dst': x_shape_padded,
    }


def pad_fwd_plan(x, plan):
    # TODO: Improve when F.pad accepts dim argument
    permute_roll = list(range(1, x.dim())) + [0]
    for i in range(x.dim()):
        x = x.permute(permute_roll)
        m, p = x.shape[-1], plan['paddings'][i]
        if p == 0:
            continue
        shape_invariant = x.shape[:-1]
        x = x.reshape(-1, 1, m)
        x = F.pad(x, [0, p], 'replicate')
        x = x.reshape(*shape_invariant, m+p)
    return x


def tensor_truncate_shape(x, shape):
    slices = [slice(a) for a in shape]
    return x[slices]


def pad_inv_plan(x, plan):
    return tensor_truncate_shape(x, plan['shape_src'])


def compute_conv_weight_reshape_to_qtt_plan(x_shape, group_tuples, squash_filters):
    """
    :param x_shape: conv weight tensor of shape [ out_channels, in_channels // groups, *kernel_size ]
    :param group_tuples: merge neighbor factors in the input-output channels factors permutation
    :param squash_filters: merge spatial kernel dimensions
    :return: transformation plan
    """
    assert len(x_shape) >= 2
    plan_channels = compute_z_order(x_shape[0:2], group_tuples=group_tuples)
    if len(x_shape[2:]) > 0:
        plan_spatial = compute_z_order(x_shape[2:], group_tuples=squash_filters)
        plan = combine_z_order_plans([plan_channels, plan_spatial])
    else:
        plan = plan_channels
    return plan


def combine_z_order_plans(plans):
    out = plans[0].copy()
    for plan in plans[1:]:
        offset = len(out['permute_factors_src_to_dst'])
        out['shape_src'] += plan['shape_src']
        out['shape_factors_src'] += plan['shape_factors_src']
        out['permute_factors_src_to_dst'] += [a + offset for a in plan['permute_factors_src_to_dst']]
        out['permute_factors_dst_to_src'] += [a + offset for a in plan['permute_factors_dst_to_src']]
        out['shape_factors_dst'] += plan['shape_factors_dst']
        out['shape_dst'] += plan['shape_dst']
    return out


def reshape_fwd_plan(x, plan):
    x = x.view(plan['shape_factors_src'])
    x = x.permute(plan['permute_factors_src_to_dst'])
    x = x.reshape(plan['shape_dst'])
    return x


def reshape_inv_plan(x, plan):
    x = x.view(plan['shape_factors_dst'])
    x = x.permute(plan['permute_factors_dst_to_src'])
    x = x.reshape(plan['shape_src'])
    return x
