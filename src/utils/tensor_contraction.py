from itertools import accumulate

import opt_einsum
import torch
from opt_einsum.parser import get_symbol


def contract_two(el1, el2):
    # type: (Tensor, Tensor) -> Tensor
    el1_shape, el1_rank = el1.shape[:-1], el1.shape[-1]
    el2_rank, el2_shape = el2.shape[0], el2.shape[1:]
    assert el1_rank == el2_rank
    el1 = el1.reshape(-1, el1_rank)
    el2 = el2.reshape(el2_rank, -1)
    el12 = el1.mm(el2)
    out = el12.reshape(el1_shape + el2_shape)
    return out


def contract_two_batch(el1, el2):
    # type: (Tensor, Tensor) -> Tensor
    el1_batch, el1_shape, el1_rank = el1.shape[0], el1.shape[1:-1], el1.shape[-1]
    el2_batch, el2_rank, el2_shape = el2.shape[0], el2.shape[1], el2.shape[2:]
    assert el1_batch == el2_batch and el1_rank == el2_rank  # produces TracerWarning -- safe to ignore
    el1 = el1.reshape(el1_batch, -1, el1_rank)
    el2 = el2.reshape(el2_batch, el2_rank, -1)
    el12 = el1.bmm(el2)
    out = el12.reshape([el1_batch] + list(el1_shape) + list(el2_shape))
    return out


def contract_trace_outter(t):
    # type: (Tensor) -> Tensor
    assert t.shape[0] == t.shape[-1]
    if t.shape[0] > 1:
        out = torch.einsum('i...i->...', t)  # TR
    else:
        out = t.squeeze(0).squeeze(-1)  # TT
    return out


def contract_trace_outter_batch(t):
    # type: (Tensor) -> Tensor
    assert t.shape[1] == t.shape[-1]
    if t.shape[1] > 1:
        out = torch.einsum('bi...i->b...', t)  # TR
    else:
        out = t.squeeze(1).squeeze(-1)  # TT
    return out


def contract_trace_last_two(t, method='index_select'):
    # type: (Tensor) -> Tensor
    assert t.shape[-1] == t.shape[-2]
    if t.shape[-1] == 1:
        return t.squeeze(-1).squeeze(-1)
    if method == 'einsum':
        out = torch.einsum('...ii->...', t)
    elif method == 'muleye':
        eye = torch.eye(t.shape[-1], device=t.device).view([1] * (t.dim() - 2) + [t.shape[-1], t.shape[-1]])
        out = (t * eye).sum(dim=[t.dim()-1, t.dim()-2])
    elif method == 'index_select':
        ind = torch.tensor(range(0, t.shape[-1] ** 2, t.shape[-1] + 1), device=t.device)
        out = t.reshape(list(t.shape[:-2]) + [t.shape[-1] ** 2]).index_select(dim=-1, index=ind).sum(dim=-1)
    else:
        raise ValueError
    return out


def split_list_cores_equally(list_cores):
    sz = [torch.tensor(a.shape).prod().item() for a in list_cores]  # produces TracerWarning -- safe to ignore
    left_cum_sz = [0] + list(accumulate(sz))
    right_cum_sz = list(reversed(list(accumulate(reversed(sz))))) + [0]
    min_join_dist, min_join_pos, min_join_left = None, None, None
    for pos in range(0, len(list_cores), 2):
        sz_left = left_cum_sz[pos]
        sz_pos = sz[pos]
        sz_right = right_cum_sz[pos + 1]
        #assert sz_left + sz_pos + sz_right == sum(sz)
        dist_join_left = abs(sz_left + sz_pos - sz_right)
        dist_join_right = abs(sz_right + sz_pos - sz_left)
        dist_join = min(dist_join_left, dist_join_right)
        if min_join_dist is None or min_join_dist > dist_join:
            min_join_dist = dist_join
            min_join_pos = pos
            min_join_left = dist_join_left < dist_join_right
    id_split_left_inclusive = min_join_pos if min_join_left else min_join_pos - 1
    #assert id_split_left_inclusive in range(len(list_cores))
    list_left = list_cores[:id_split_left_inclusive + 1]
    list_right = list_cores[id_split_left_inclusive + 1:]
    return list_left, list_right


ABC = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'


def contract_two_ring(el1, el2):
    assert el1.shape[0] == el2.shape[-1] and el1.shape[-1] == el2.shape[0]
    dims_inner_L = ABC[2 : 2 + (el1.dim() - 2)]
    dims_inner_R = ABC[2 + (el1.dim() - 2) : 2 + (el1.dim() - 2) + (el2.dim() - 2)]
    einsum_prog = 'a' + dims_inner_L + 'b,b' + dims_inner_R + 'a->' + dims_inner_L + dims_inner_R
    out = torch.einsum(einsum_prog, el1, el2)
    return out


def contract_two_ring_batch(el1, el2):
    assert el1.shape[0] == el2.shape[0] and el1.shape[1] == el2.shape[-1] and el1.shape[-1] == el2.shape[1]  # produces TracerWarning -- safe to ignore
    dims_inner_L = ABC[3 : 3 + (el1.dim() - 3)]
    dims_inner_R = ABC[3 + (el1.dim() - 3) : 3 + (el1.dim() - 3) + (el2.dim() - 3)]
    einsum_prog = 'ba' + dims_inner_L + 'c,bc' + dims_inner_R + 'a->b' + dims_inner_L + dims_inner_R
    out = torch.einsum(einsum_prog, el1, el2)
    return out


def contract_sequence_hierarchical_one_sweep(list_cores):
    out = []
    while len(list_cores) >= 2:
        el1 = list_cores.pop(0)
        el2 = list_cores.pop(0)
        out.append(contract_two(el1, el2))
    if len(list_cores) > 0:
        el = list_cores.pop(0)
        out.append(el)
    return out


def contract_sequence_hierarchical_one_sweep_v2(list_cores):
    """
    Contracts pairs and tries to get equally-sized pair in the penultimate step, in order to apply efficient merge+trace
    """
    if len(list_cores) < 2:
        raise ValueError
    elif len(list_cores) == 2:
        out = contract_two_ring(*list_cores)
    elif len(list_cores) % 2 == 0:
        out = contract_sequence_hierarchical_one_sweep(list_cores)
    else:
        list_left, list_right = split_list_cores_equally(list_cores)
        out = contract_sequence_hierarchical_one_sweep(list_left)
        if len(list_right) % 2 == 1 and len(list_right) > 1:
            out += [list_right[0]]
            out += contract_sequence_hierarchical_one_sweep(list_right[1:])
        else:
            out += contract_sequence_hierarchical_one_sweep(list_right)
    return out


def contract_sequence_hierarchical_one_sweep_batch(list_cores):
    out = []
    while len(list_cores) >= 2:
        el1 = list_cores.pop(0)
        el2 = list_cores.pop(0)
        out.append(contract_two_batch(el1, el2))
    if len(list_cores) > 0:
        el = list_cores.pop(0)
        out.append(el)
    return out


def contract_sequence_hierarchical_one_sweep_batch_v2(list_cores):
    if len(list_cores) < 2:
        raise ValueError
    elif len(list_cores) == 2:
        out = contract_two_ring_batch(*list_cores)
    elif len(list_cores) % 2 == 0:
        out = contract_sequence_hierarchical_one_sweep_batch(list_cores)
    else:
        list_left, list_right = split_list_cores_equally(list_cores)
        out = contract_sequence_hierarchical_one_sweep_batch(list_left)
        if len(list_right) % 2 == 1 and len(list_right) > 1:
            out += [list_right[0]]
            out += contract_sequence_hierarchical_one_sweep_batch(list_right[1:])
        else:
            out += contract_sequence_hierarchical_one_sweep_batch(list_right)
    return out


def contract_sequence_hierarchical(*list_cores):
    list_cores = list(list_cores)
    while True:
        list_cores = contract_sequence_hierarchical_one_sweep(list_cores)
        if len(list_cores) == 1:
            break
    return list_cores[0]


def contract_sequence_hierarchical_batch(*list_cores):
    list_cores = list(list_cores)
    while True:
        list_cores = contract_sequence_hierarchical_one_sweep_batch(list_cores)
        if len(list_cores) == 1:
            break
    return list_cores[0]


def contract_composition_hierarchical(*list_cores):
    out = contract_sequence_hierarchical(*list_cores)
    out = contract_trace_outter(out)
    return out


def contract_composition_hierarchical_batch(*list_cores):
    out = contract_sequence_hierarchical_batch(*list_cores)
    out = contract_trace_outter_batch(out)
    return out


def contract_composition_hierarchical_batch_v2(*list_cores):
    list_cores = list(list_cores)
    while True:
        list_cores = contract_sequence_hierarchical_one_sweep_batch_v2(list_cores)
        if type(list_cores) is not list:
            break
    return list_cores


def contract_composition_sequential(*list_cores):
    list_cores = list(list_cores)
    out = list_cores[0]
    for core in list_cores[1:]:
        out = contract_two(out, core)
    out = contract_trace_outter(out)
    return out


def compute_contraction_fn(core_shapes):
    assert all([len(shape) == 3 for shape in core_shapes])  # produces TracerWarning -- safe to ignore
    next_sym = 0
    def next():
        nonlocal next_sym
        s = get_symbol(next_sym)
        next_sym += 1
        return s
    equation_left = ""
    equation_right = ""
    letter_core0_rank_left = None
    letter_core_last_rank_right = None
    for i in range(len(core_shapes)):
        if i == 0:
            letter_rank_left = next()
            letter_core0_rank_left = letter_rank_left
        else:
            letter_rank_left = letter_core_last_rank_right
        letter_mode = next()
        if i == len(core_shapes) - 1:
            letter_rank_right = letter_core0_rank_left
        else:
            letter_rank_right = next()
            letter_core_last_rank_right = letter_rank_right
        if i > 0:
            equation_left += ','
        equation_left += letter_rank_left
        equation_left += letter_mode
        equation_left += letter_rank_right
        equation_right += letter_mode
    equation = equation_left + '->' + equation_right
    contraction_fn = opt_einsum.contract_expression(equation, *core_shapes, optimize='dp')
    return contraction_fn


def resolve_contraction_fn(method, core_shapes):
    if method == 'opt_einsum':
        return compute_contraction_fn(core_shapes)
    elif method == 'hierarchical':
        return contract_composition_hierarchical
    elif method == 'sequential':
        return contract_composition_sequential
    else:
        raise NotImplementedError(f'Unknown tensor contraction method {method}')
