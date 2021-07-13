import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.init import _calculate_correct_fan, calculate_gain

from src.utils.tensor_contraction import resolve_contraction_fn, contract_sequence_hierarchical_one_sweep_v2, \
    contract_composition_hierarchical_batch_v2
from src.utils.tensor_shape_tools import compute_pad_to_powers_plan, tensor_truncate_shape, reshape_inv_plan, \
    pad_inv_plan, combine_z_order_plans, compute_z_order


class TBasis(torch.nn.Module):
    def __init__(self, size, rank, mode, is_trainable, init_seed):
        super().__init__()
        self.size = size
        self.rank = rank
        self.mode = mode
        self.is_trainable = is_trainable
        self.num_param = 0
        self.rng = np.random if init_seed is None else np.random.RandomState(init_seed)
        with torch.no_grad():
            init = torch.randn((self.size, self.rank, self.mode, self.rank)).clamp(-3, 3)
            init /= math.sqrt(size * rank)
        if is_trainable:
            self.basis = torch.nn.Parameter(init)
            self.num_param = self.size * self.rank * self.mode * self.rank
        else:
            self.register_buffer('basis', init)
        self.last_basis = None

    def forward(self):
        self.last_basis = self.basis
        return self.last_basis

    def num_parameters(self):
        return self.num_param

    def extra_repr(self):
        return f'[{self.size} x {self.rank} x {self.mode} x {self.rank}] ' \
            f'{"trainable" if self.is_trainable else "fixed"}'
    
    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        sd.pop('last_basis', None)
        return sd


class RankAdaptationModule(torch.nn.Module):
    def __init__(self, num_modes, rank, activation):
        super().__init__()
        assert num_modes > 0
        assert activation in ('linear', 'exp')
        self.num_modes = num_modes
        self.rank = rank
        self.activation = activation
        self.rank_adapters = torch.nn.Parameter(torch.Tensor(num_modes, rank))
        self._num_parameters = num_modes * rank
        self.initialize_identity()

    def forward(self):
        return self._act_fwd(self.rank_adapters, self.activation)

    @staticmethod
    def _act_fwd(x, activation):
        return {
            'linear': lambda a: a,
            'exp': torch.exp,
        }[activation](x)

    @staticmethod
    def _act_inv(x, activation):
        return {
            'linear': lambda a: a,
            'exp': torch.log,
        }[activation](x)

    def initialize_const(self, c):
        init = torch.full((self.num_modes, self.rank), fill_value=c, dtype=torch.float32)
        init = self._act_inv(init, self.activation)
        with torch.no_grad():
            self.rank_adapters.copy_(init)

    def initialize_identity(self):
        self.initialize_const(1)

    def num_parameters(self):
        return self._num_parameters

    def extra_repr(self):
        return f'[{self.rank} x {self.num_modes}] activation={self.activation}'


class TBasisWeightsFactory(torch.nn.Module):
    def __init__(
            self,
            basis_size,
            basis_rank,
            rank_adaptation,
            forward_batch_size=0,
    ):
        """
        Parameterization of all neural network weights through T-Basis of a certain configuration
        :param basis_size: number of basis cores
        :param basis_rank: rank of basis cores
        :param rank_adaptation: whether to perform rank adaptation (adds extra learned parameters)
        :param forward_batch_size: whether to perform batching of cores during full tensors assembly. 0 = max possible
        """
        super().__init__()
        self.basis_size = basis_size
        self.basis_rank = basis_rank
        self.rank_adaptation = rank_adaptation
        self.forward_batch_size = forward_batch_size
        self.num_modes_total = 0
        self.list_names = []
        self.list_modes_offset = []
        self.list_modes_count = []
        self.list_init_scaling_factors = []
        self.weights = None  # will be torch.nn.Parameter of size [ num_modes_total x basis_size ]
        self.rank_adapters = None  # will be torch.nn.Parameter of size [ num_modes_total x basis_rank ]
        self.plan_batching = None

    def add_tensor(self, name, num_modes, init_scaling_factor):
        self.list_names.append(name)
        self.list_modes_offset.append(self.num_modes_total)
        self.list_modes_count.append(num_modes)
        self.list_init_scaling_factors.append(init_scaling_factor)
        self.num_modes_total += num_modes

    def instantiate_parameters(self, plan_batching=None):
        assert self.num_modes_total > 0
        self.plan_batching = plan_batching

        weights = []
        for name, modes_count, init_scaling_factor in \
                zip(self.list_names, self.list_modes_count, self.list_init_scaling_factors):
            rng_seed = hash(name) % (2 ** 32)
            rng = np.random.RandomState(rng_seed)
            w = torch.from_numpy(rng.randn(modes_count, self.basis_size)).float().clamp(-3, 3) * init_scaling_factor
            weights.append(w)
        weights = torch.cat(weights, dim=0)

        if self.plan_batching is not None and (self.forward_batch_size == 0 or self.forward_batch_size > 1):
            weights = weights[plan_batching['map_core_ids_flat_to_grouped_by_batch'], :]

        self.weights = torch.nn.Parameter(weights)
        if self.rank_adaptation is not None:
            rankadapt = torch.ones((self.num_modes_total, self.basis_rank), dtype=torch.float32)
            rankadapt = RankAdaptationModule._act_inv(rankadapt, self.rank_adaptation)
            self.rank_adapters = torch.nn.Parameter(rankadapt)

    def forward(self, tbasis):
        # weights : D x B
        # tbasis   : B x R x M x R
        ntensors = len(self.list_names)
        core_shape = tbasis.shape[1:]
        tbasis = tbasis.view(self.basis_size, -1)
        cores = self.weights.mm(tbasis).view(self.num_modes_total, *core_shape)  # D x R x M x R
        if self.rank_adaptation is not None:
            rank_adapters = RankAdaptationModule._act_fwd(self.rank_adapters, self.rank_adaptation)
            cores = cores * rank_adapters.view(self.num_modes_total, self.basis_rank, 1, 1)  # D x R x 1 x 1
        # D x R x M x R

        if self.forward_batch_size == 1:
            cores = [core.squeeze(0) for core in cores.chunk(self.num_modes_total, dim=0)]
            it_cores = iter(cores)
            cores = [[next(it_cores) for _ in range(size)] for i, size in enumerate(self.list_modes_count)]

        if self.forward_batch_size == 1:
            # This special case saves memory on unnecessary order permutations
            num_decompressed = 0
            while True:
                if num_decompressed == ntensors:
                    break
                for i in range(ntensors):
                    if type(cores[i]) is not list:
                        continue
                    cores[i] = contract_sequence_hierarchical_one_sweep_v2(cores[i])
                    if type(cores[i]) is not list:
                        num_decompressed += 1
            return tuple(cores)

        # case of self.forward_batch_size == 0 (max possible batching) or >1
        list_tensors = []
        last_core_id = 0
        for tensor_modes, tensor_modes_counts in \
                zip(self.plan_batching['unique_tensor_modes'], self.plan_batching['unique_tensor_modes_counts']):
            batched_cores = []
            for mode_id in range(tensor_modes):
                batched_core = cores[last_core_id:last_core_id+tensor_modes_counts, :, :, :]
                assert batched_core.dim() == 4
                batched_cores.append(batched_core)
                last_core_id += tensor_modes_counts

            if self.forward_batch_size > 1:
                minibatch_offset = 0
                for i in range((tensor_modes_counts + self.forward_batch_size - 1) // self.forward_batch_size):
                    minibatched_cores = [
                        a[minibatch_offset:minibatch_offset + self.forward_batch_size] for a in batched_cores
                    ]
                    minibatch_tensors = contract_composition_hierarchical_batch_v2(*minibatched_cores)
                    assert minibatch_tensors.dim() == tensor_modes + 1
                    minibatch_tensors = [
                        t.squeeze(0) for t in minibatch_tensors.chunk(minibatch_tensors.shape[0], dim=0)
                    ]
                    list_tensors.extend(minibatch_tensors)
                    minibatch_offset += self.forward_batch_size
            else:
                tensors = contract_composition_hierarchical_batch_v2(*batched_cores)
                assert tensors.dim() == tensor_modes + 1
                tensors = [t.squeeze(0) for t in tensors.chunk(tensors.shape[0], dim=0)]
                list_tensors.extend(tensors)

        list_tensors = tuple(
            list_tensors[self.plan_batching['map_tensors_batched_to_original'][a]] for a in range(len(self.list_names))
        )
        return list_tensors

    @staticmethod
    def create_batching_plan(list_tensor_modes):
        # list_tensor_modes: [ 3, 4, 4, 4, 3, 2 ]
        num_modes_total = sum(list_tensor_modes)
        list_tensor_modes_argsorted = list(np.argsort(list_tensor_modes))
        # [ 5, 0, 4, 1, 2, 3 ]
        unique_tensor_modes, unique_tensor_modes_counts = np.unique(list_tensor_modes, return_counts=True)
        # unique_tensor_modes:        [ 2, 3, 4 ]
        # unique_tensor_modes_counts: [ 1, 2, 3 ]
        iter_ids = iter(range(num_modes_total))
        core_ids = [[next(iter_ids) for _ in range(num_modes)] for num_modes in list_tensor_modes]
        # [ [0,1,2], [3,4,5,6], [7,8,9,10], [11,12,13,14], [15,16,17], [18,19] ]
        core_ids_sorted_by_group_size = [core_ids[i] for i in list_tensor_modes_argsorted]
        # [ [18,19], [0,1,2], [15,16,17], [3,4,5,6], [7,8,9,10], [11,12,13,14] ]
        map_core_ids_flat_to_grouped_by_size = [b for a in core_ids_sorted_by_group_size for b in a]
        # [ 18, 19, 0, 1, 2, 15, 16, 17, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 ]
        map_core_ids_grouped_by_size_to_flat = list(np.argsort(map_core_ids_flat_to_grouped_by_size))
        # [ 2, 3, 4, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 5, 6, 7, 0, 1 ]
        map_core_ids_grouped_by_batch_to_size = []
        last_core_id = 0
        for i in range(len(unique_tensor_modes)):
            num_modes_in_group = unique_tensor_modes_counts[i] * unique_tensor_modes[i]
            grouped_by_size_core_ids = np.arange(last_core_id, last_core_id + num_modes_in_group)
            last_core_id += num_modes_in_group
            grouped_by_batch_core_ids = grouped_by_size_core_ids.reshape(
                unique_tensor_modes[i], unique_tensor_modes_counts[i]
            ).T.reshape(-1)
            map_core_ids_grouped_by_batch_to_size.extend(list(grouped_by_batch_core_ids))
        # [0, 1, 2, 4, 6, 3, 5, 7, 8, 11, 14, 17, 9, 12, 15, 18, 10, 13, 16, 19]
        map_core_ids_grouped_by_size_to_batch = list(np.argsort(map_core_ids_grouped_by_batch_to_size))
        # [0, 1, 2, 5, 3, 6, 4, 7, 8, 12, 16, 9, 13, 17, 10, 14, 18, 11, 15, 19]
        map_core_ids_flat_to_grouped_by_batch = [
            map_core_ids_flat_to_grouped_by_size[a] for a in map_core_ids_grouped_by_size_to_batch
        ]
        # [ 18, 19, 0, 15, 1, 16, 2, 17, 3, 7, 11, 4, 8, 12, 5, 9, 13, 6, 10, 14 ]
        map_core_ids_grouped_by_batch_to_flat = list(np.argsort(map_core_ids_flat_to_grouped_by_batch))
        # [ 2, 4, 6, 8, 11, 14, 17, 9, 12, 15, 18, 10, 13, 16, 19, 3, 5, 7, 0, 1 ]
        return {
            'unique_tensor_modes': unique_tensor_modes,
            'unique_tensor_modes_counts': unique_tensor_modes_counts,
            'map_tensors_batched_to_original': list(np.argsort(list_tensor_modes_argsorted)),
            'map_core_ids_flat_to_grouped_by_batch': map_core_ids_flat_to_grouped_by_batch,
            'map_core_ids_grouped_by_batch_to_flat': map_core_ids_grouped_by_batch_to_flat
        }

    def _get_tbasis_parameter_flat_order(self, param):
        if self.plan_batching is not None and (self.forward_batch_size == 0 or self.forward_batch_size > 1):
            param = param[self.plan_batching['map_core_ids_grouped_by_batch_to_flat'], :]
        list_param = []
        for offset, count in zip(self.list_modes_offset, self.list_modes_count):
            list_param.append(param[offset : offset + count, :])
        return list_param

    def get_tbasis_weights(self):
        return dict(zip(self.list_names, self._get_tbasis_parameter_flat_order(self.weights)))

    def get_tbasis_rank_adapters(self):
        if self.rank_adaptation is None:
            return None
        rank_adapters = RankAdaptationModule._act_fwd(self.rank_adapters, self.rank_adaptation)
        return dict(zip(self.list_names, self._get_tbasis_parameter_flat_order(rank_adapters)))


class TBasisWeightsFactoryProxy:
    """
    While TBasisWeightsFactory may be subject to JIT, it incurs certain constraints on accessing JIT class state.
    This class will be mediating TBasisWeightsFactory.forward() output by means of set_tbasis_coefficients and
    get_coefficients_by_name functions to all the convolutional and linear modules.
    """
    def __init__(self, tbasis_weights_factory):
        self.tbasis_weights_factory = tbasis_weights_factory
        self.list_names = []
        self.map_name_to_id = {}
        self.plan_batching = None
        self.last_tensors = None
        self.list_modes_count = []

    def add_tensor(self, name, num_modes, init_scaling_factor):
        assert name not in self.map_name_to_id
        self.map_name_to_id[name] = len(self.list_names)
        self.list_names.append(name)
        self.list_modes_count.append(num_modes)
        self.tbasis_weights_factory.add_tensor(name, num_modes, init_scaling_factor)

    def instantiate_parameters(self):
        self.plan_batching = TBasisWeightsFactory.create_batching_plan(self.list_modes_count)
        self.tbasis_weights_factory.instantiate_parameters(self.plan_batching)

    def set_tbasis_tensors(self, tensors):
        self.last_tensors = tensors

    def get_tensor_by_name(self, name):
        return self.last_tensors[self.map_name_to_id[name]]

    def get_tbasis_weights(self):
        return self.tbasis_weights_factory.get_tbasis_weights()

    def get_tbasis_rank_adapters(self):
        return self.tbasis_weights_factory.get_tbasis_rank_adapters()


class TBasisLayerBase(torch.nn.Module):
    def __init__(
            self,
            module,
            basis_module,
            basis_tensors_proxy,
            contraction_method,
            rank_adaptation=None,
            permute_and_group_factors=True,
            use_external_parameterization=True,
            sanity_check_once=False,
            verbose=False,
            name=None,
    ):
        super().__init__()
        self.basis_module = [basis_module]  # list prevents multiple occurrences in torch.nn.Module.parameters()
        self.basis_tensors_proxy = basis_tensors_proxy
        self.name = name
        self.num_basis = basis_module.size
        self.rank = basis_module.rank
        self.mode = basis_module.mode
        self.rank_adaptation = rank_adaptation
        self.permute_and_group_factors = permute_and_group_factors
        self.use_external_parameterization = use_external_parameterization
        self.sanity_check_once = sanity_check_once
        self.init_checks(module)
        self.mode_base = int(math.sqrt(self.mode)) if permute_and_group_factors else self.mode
        self.plan_pad = compute_pad_to_powers_plan(module.weight.shape, self.mode_base)
        weight_shape_padded = self.plan_pad['shape_dst']
        weight_is_linear = len(module.weight.shape) == 2
        assert len(weight_shape_padded) >= 2
        self.plan_shape_channels = compute_z_order(
            weight_shape_padded[0:2], group_tuples=self.permute_and_group_factors
        )
        if weight_is_linear:
            self.plan_shape = self.plan_shape_channels
        else:
            self.plan_shape_spatial = compute_z_order(
                weight_shape_padded[2:], group_tuples=self.permute_and_group_factors
            )
            self.plan_shape = combine_z_order_plans([self.plan_shape_channels, self.plan_shape_spatial])
        self.num_modes = len(self.plan_shape['shape_dst'])
        assert all(a in (self.mode, self.mode_base) for a in self.plan_shape['shape_dst'])
        # rng seeding is here to synchronize weights used here and in TBasisTensors with respect to the scaler
        rng_seed = hash(name) % (2 ** 32)
        rng = np.random.RandomState(rng_seed)
        basis_weights = torch.from_numpy(rng.randn(self.num_modes, self.num_basis)).float().clamp(-3, 3)
        self.basis_weights = torch.nn.Parameter(basis_weights)
        self.num_param = self.num_modes * self.num_basis
        if rank_adaptation is not None:
            self.rank_adapters = RankAdaptationModule(self.num_modes, self.rank, rank_adaptation)
            self.num_param += self.rank_adapters.num_parameters()
        self.contraction_fn = resolve_contraction_fn(
            contraction_method,
            [(self.rank, self.mode, self.rank) for _ in range(self.num_modes)]
        )
        if module.bias is None:
            self.register_parameter('bias', None)
        else:
            self.bias = module.bias
        self.initialize(verbose=verbose)

    def initialize(self, fan_mode='fan_in', nonlinearity='relu', nonlinearity_negative_slope=0, verbose=False):
        with torch.no_grad():
            fan = _calculate_correct_fan(torch.empty(*self.plan_pad['shape_src']), fan_mode)
            gain = calculate_gain(nonlinearity, nonlinearity_negative_slope)
            std_target = gain / math.sqrt(fan)
            _ = self.basis_module[0].forward()  # update basis_module.last_basis
            std_decompressed_before_correction = self.decompress_full(
                override_use_external_parameterization=False
            ).std().item()
            std_correction = std_target / std_decompressed_before_correction
            core_scaler = std_correction ** (1 / self.num_modes)
            self.basis_weights *= core_scaler
            std_decompressed_after_correction = self.decompress_full(
                override_use_external_parameterization=False
            ).std().item()
            if verbose:
                msg = \
                    f'Init {self.layer_type_name if self.name is None else self.name:64} ' \
                    f'std_before_corr={std_decompressed_before_correction:6.4f} ' \
                    f'scaler={core_scaler:6.4f} ' \
                    f'std_after_corr={std_decompressed_after_correction:6.4f} ' \
                    f'std_target={std_target:6.4f} '
                print(msg)
            if self.use_external_parameterization:
                self.basis_tensors_proxy.add_tensor(self.name, self.num_modes, core_scaler)
                if not self.sanity_check_once:
                    del self.basis_weights
                    if self.rank_adaptation is not None:
                        del self.rank_adapters

    def decompress_cores(self):
        basis_weights = self.basis_weights  # D x B
        basis = self.basis_module[0].last_basis  # B x R x M x R
        core_shape = basis.shape[1:]
        basis = basis.view(self.num_basis, -1)
        cores = basis_weights.mm(basis).view(self.num_modes, *core_shape)  # D x R x M x R
        if self.rank_adaptation is not None:
            cores = cores * self.rank_adapters().view(self.num_modes, self.rank, 1, 1)
        list_cores = cores.chunk(self.num_modes, dim=0)
        list_cores = [core.squeeze(0) for core in list_cores]
        return list_cores

    def decompress_full(self, override_use_external_parameterization=None):
        use_external_parameterization = self.use_external_parameterization
        if override_use_external_parameterization is not None:
            use_external_parameterization = override_use_external_parameterization
        if use_external_parameterization:
            weight = self.basis_tensors_proxy.get_tensor_by_name(self.name)
            if self.sanity_check_once:
                list_cores = self.decompress_cores()
                weight_old = self.contraction_fn(*list_cores)
                residual = (weight_old - weight).abs().max().item()
                raise Exception(f'Max abs diff between external and internal parameterizations: {residual}')
        else:
            list_cores = self.decompress_cores()
            weight = self.contraction_fn(*list_cores)
        if self.permute_and_group_factors:
            weight = tensor_truncate_shape(weight, self.plan_shape['shape_dst'])
        weight = reshape_inv_plan(weight, self.plan_shape)
        weight = pad_inv_plan(weight, self.plan_pad)
        return weight

    def num_parameters(self):
        return self.num_param


class TBasisConvNd(TBasisLayerBase):
    map_type_to_name = {
        torch.nn.Conv1d: 'Conv1d',
        torch.nn.Conv2d: 'Conv2d',
        torch.nn.Conv3d: 'Conv3d',
    }

    map_type_to_F = {
        torch.nn.Conv1d: F.conv1d,
        torch.nn.Conv2d: F.conv2d,
        torch.nn.Conv3d: F.conv3d,
    }

    def __init__(self, module, *args, **kwargs):
        self.layer_type_name = self.map_type_to_name[type(module)]
        super().__init__(module, *args, **kwargs)
        self.conv_fn = self.map_type_to_F[type(module)]
        self.in_channels = module.in_channels
        self.out_channels = module.out_channels
        self.kernel_size = module.kernel_size
        self.kernel_numel = torch.tensor(self.kernel_size).prod().item()
        self.stride = module.stride
        self.padding = module.padding
        self.dilation = module.dilation
        self.groups = module.groups
        self.input_unfolder = torch.nn.Unfold(
            module.kernel_size,
            dilation=module.dilation,
            padding=module.padding,
            stride=module.stride
        )

    def init_checks(self, module):
        assert isinstance(module, torch.nn.modules.conv._ConvNd)
        assert not isinstance(module, torch.nn.modules.conv._ConvTransposeMixin), 'Not implemented'
        assert module.padding_mode != 'circular', 'Not implemented'

    def forward(self, input):
        """
        Decompress TR-weight into a regular weight and apply torch convolution.
        """
        weight = self.decompress_full()
        return self.conv_fn(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def extra_repr(self):
        return f'[{" x ".join([str(a) for a in self.plan_pad["shape_src"]])}] stride={self.stride} ' \
            f'padding={self.padding} dilation={self.dilation} groups={self.groups} has_bias={self.bias is not None}'


class TBasisLinear(TBasisLayerBase):
    def __init__(self, *args, **kwargs):
        self.layer_type_name = 'Linear'
        super().__init__(*args, **kwargs)

    def init_checks(self, module):
        assert isinstance(module, torch.nn.modules.Linear)

    def forward(self, input):
        weight = self.decompress_full()
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return f'[{" x ".join([str(a) for a in self.plan_pad["shape_src"]])}] has_bias={self.bias is not None}'
