import torch

from src.utils.helpers import get_module_num_params, deep_transform, get_statedict_num_params, get_zipped_size
from src.utils.tensorized_basis_modules_tr import TBasisConvNd, TBasisLinear


def net_prepare_training(net, basis_module, basis_tensors_proxy, cfg, net_prefix=None):

    def cb_compress(module, prefix, opaque):
        sz_module_total = get_module_num_params(module, recurse=False, include_buffers=True)
        sz_module_param = get_module_num_params(module, recurse=False, include_buffers=False)
        opaque['size_original_total'] += sz_module_total
        opaque['size_original_param'] += sz_module_param

        def mark_module_unprocessed():
            opaque['size_incompressible_total'] += sz_module_total
            opaque['size_incompressible_param'] += sz_module_param

        if prefix in cfg.module_names_ignored:
            mark_module_unprocessed()
            return module

        if isinstance(module, torch.nn.modules.conv._ConvNd):
            replacement_cls = TBasisConvNd
        elif isinstance(module, torch.nn.Linear):
            replacement_cls = TBasisLinear
        elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            opaque['size_incompressible_total'] += get_module_num_params(module, recurse=True, include_buffers=True)
            opaque['size_incompressible_param'] += get_module_num_params(module, recurse=True, include_buffers=False)
            return module
        else:
            mark_module_unprocessed()
            return module

        replacement_module = replacement_cls(
            module,
            basis_module,
            basis_tensors_proxy,
            contraction_method=cfg.contraction_method,
            rank_adaptation=cfg.rank_adaptation,
            permute_and_group_factors=cfg.permute_and_group_factors,
            verbose=True,
            name=prefix,
        )

        sz_weight_unparam = torch.tensor(module.weight.shape).prod().item()
        sz_weight_param = replacement_module.num_parameters()
        sz_bias = module.bias.numel() if module.bias is not None else 0

        if sz_weight_param > sz_weight_unparam:
            print(f'Module {prefix} original size is smaller than compressed: {sz_weight_unparam} < {sz_weight_param}')

        opaque['size_incompressible_total'] += sz_bias
        opaque['size_incompressible_param'] += sz_bias
        opaque['size_compressible_unparameterized'] += sz_weight_unparam
        opaque['size_compressible_parameterized'] += sz_weight_param

        return replacement_module

    out = {
        'size_original_total': 0,
        'size_original_param': 0,
        'size_incompressible_total': 0,
        'size_incompressible_param': 0,
        'size_compressible_unparameterized': 0,
        'size_compressible_parameterized': 0,
    }

    szcheck_original_total = get_statedict_num_params(net.state_dict())

    deep_transform(net, cb_compress, prefix=net_prefix, opaque=out)
    basis_tensors_proxy.instantiate_parameters()

    sz_basis = basis_module.num_parameters()
    sz_coef = get_statedict_num_params(basis_tensors_proxy.tbasis_weights_factory.state_dict())

    out['size_basis'] = sz_basis
    out['size_coef'] = sz_coef

    out['compression_compressible'] = \
        100 * out['size_compressible_parameterized'] / out['size_compressible_unparameterized']
    out['compression_limit_total'] = \
        100 * out['size_incompressible_total'] / out['size_original_total']
    out['compression_limit_param'] = \
        100 * out['size_incompressible_param'] / out['size_original_param']
    out['compression_net_total_without_basis'] = \
        100 * (out['size_compressible_parameterized'] + out['size_incompressible_total']) / out['size_original_total']
    out['compression_net_total_with_basis'] = \
        out['compression_net_total_without_basis'] + 100 * sz_basis / out['size_original_total']
    out['compression_net_param_without_basis'] = \
        100 * (out['size_compressible_parameterized'] + out['size_incompressible_param']) / out['size_original_param']
    out['compression_net_param_with_basis'] = \
        out['compression_net_param_without_basis'] + 100 * sz_basis / out['size_original_param']

    szcheck_basis = get_statedict_num_params(basis_module.state_dict())
    szcheck_incompressible_total = get_statedict_num_params(net.state_dict())
    compressioncheck_net_total_with_basis = 100 * (sz_basis + sz_coef + szcheck_incompressible_total) / szcheck_original_total
    compressioncheck_net_total_without_basis = 100 * (sz_coef + szcheck_incompressible_total) / szcheck_original_total

    assert abs(out['compression_net_total_with_basis'] - compressioncheck_net_total_with_basis) < 0.001
    assert abs(out['compression_net_total_without_basis'] - compressioncheck_net_total_without_basis) < 0.001
    assert abs(out['size_incompressible_total'] - szcheck_incompressible_total) < 0.001
    assert abs(sz_basis - szcheck_basis) < 0.001
    assert abs(out['size_original_total'] - szcheck_original_total) < 0.001

    return out
