import torch.nn as nn
import numpy
import torch


class QuantOp():
    def __init__(self, model, scale=1.0, threshold=0.01, quant_conv='mix-pow2'):
        self.scale = scale
        self.threshold = threshold
        self.scales = None

        self.saved_params = []
        self.saved_maskfix = []
        self.saved_maskfloat = []
        self.target_modules = []

        for name, m in model.named_modules():
            if not (isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear)):
                continue
            print(f'{name}\t{tuple(m.weight.size())}')
            self.saved_params.append(m.weight.data.clone())
            self.target_modules.append(m)

        self.quantizeConv = self._getQuantFunc(quant_conv)

        self.num_module = len(self.target_modules)

    def _getQuantFunc(self, quant_conv):
        if quant_conv in ['int4', 'single-pow2', 'mix-pow2', 'ex-mix-pow2']:

            quant_values_dict = {
                0: [0, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 9, -9, 10, -10],
                1: [0, 3, -3, 4, -4, 5, -5, 6, -6, 9, -9, 10, -10, 17, -17],
                2: [0, 4, -4, 6, -6, 8, -8, 10, -10, 12, -12, 18, -18, 20, -20],
                3: [0, 3, -3, 4, -4, 5, -5, 6, -6, 8, -8, 10, -10, 12, -12],
            }

            self.quant_values = {
                'int4': [0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 7, -7],
                'single-pow2': [0, 1, -1, 2, -2, 4, -4, 8, -8, 16, -16, 32, -32, 64, -64],
                'mix-pow2': [0, 2, -2, 3, -3, 4, -4, 5, -5, 6, -6, 9, -9, 10, -10],
                'ex-mix-pow2': list(set(v for vs in quant_values_dict.values() for v in vs))
            }[quant_conv]

            self.scales = [self._getScalingFactor(m) for m in self.target_modules]

            if quant_conv == 'ex-mix-pow2':
                self.q_values_list = None
                self.quant_values_dict = quant_values_dict
                return self.quantizeConvParamsExMixedPow2

            return self.quantizeConvParamsProjection

        if quant_conv == '8bit':
            return self.quantize8bit

        raise ValueError(f"invalid quant method: {quant_conv}")

    def _getScalingFactor(self, module):
        scales = numpy.linspace(0.001, 0.05, 50)

        best_scale, min_dist = 0, float("inf")
        for scale in scales:
            module_clone = module.weight.clone().detach()
            quant_values = [scale * v for v in self.quant_values]
            self._doProjection(module_clone, quant_values)
            dist = module_clone.sub(module.weight).pow(2).sum().item()  # TODO l2 norm
            if dist < min_dist:
                best_scale, min_dist = scale, dist

        print('found', best_scale)
        module.scale.data.fill_(best_scale)
        return best_scale

    def save_params(self):
        for index in range(self.num_module):
            self.saved_params[index].copy_(self.target_modules[index].weight.data)

    def _doProjection(self, module, quant_values):
        with torch.no_grad():
            # calculate absolute distance between each weight and each quant_value
            dist = torch.stack([module.sub(v).abs()
                                for v in quant_values], dim=module.dim())
            # grouping using shortest abs distance
            group_idx = dist.argmin(dim=module.dim())
            # get indice for each group
            idxs = [group_idx.data == v for v in range(len(quant_values))]
            # set the quantization value for each group
            for idx, quant_value in zip(idxs, quant_values):
                module.data[idx] = quant_value

    def quantize8bit(self, **kwargs):
        with torch.no_grad():
            for module in self.target_modules:
                module.weight.data = activation_quantization(module.weight)[0]

    def quantizeConvParamsProjection(self, q_values_list=None, **kwargs):
        if q_values_list is None:
            q_values_list = [self.quant_values] * self.num_module

        assert len(self.target_modules) == len(
            q_values_list) == len(self.scales)

        for module, scale, q_values in zip(self.target_modules, self.scales, q_values_list):
            q_values = [scale * value for value in q_values]
            self._doProjection(module.weight, q_values)

    def quantizeConvParamsExMixedPow2(self, seperate=False, **kwargs):
        # TODO Haven't check correctness
        def fix_quant_values_foreach_module():

            def totalMatch(module, q_values, scale):
                q_values = [scale * v for v in q_values]
                return sum((module.data == v).sum() for v in q_values)

            self.q_values_list = []
            for module, scale in zip(self.target_modules, self.scales):
                pairs = [
                    (key, totalMatch(module.weight, q_values, scale))
                    for key, q_values in self.quant_values_dict.items()
                ]
                best_key, _ = max(pairs, key=lambda pair: pair[1])
                print(f'key: {best_key}   match: {_}')
                self.q_values_list.append(self.quant_values_dict[best_key])

        if seperate:
            if self.q_values_list is None:
                self.quantizeConvParamsProjection()
                fix_quant_values_foreach_module()
                self.restore()

        return self.quantizeConvParamsProjection(self.q_values_list)

    def quantization(self, **kwargs):
        self.save_params()
        centers = self.quantizeConv(**kwargs)
        return centers

    def restore(self):
        for index in range(self.num_module):
            self.target_modules[index].weight.data.copy_(self.saved_params[index])

    def inq_quantization(self, threshold=1.0):

        self.saved_maskfix = []
        self.saved_maskfloat = []
        for index in range(self.num_module):
            qw_abs = self.target_modules[index].weight.data.abs()

            num_fix = int(threshold * qw_abs.nelement())
            qw_abs_topk, fix_index = qw_abs.topk(num_fix)

            # TODO use fix_index instead
            mask_fix = qw_abs.ge(qw_abs_topk.min())
            mask_float = qw_abs.lt(qw_abs_topk.min())
            self.saved_maskfix.append(mask_fix)
            self.saved_maskfloat.append(mask_float)

        self.save_params()  # TODO save float_index only
        self.quantizeConv()

        # Restore float_index only (keep fix_index fixed)
        for module, float_index, saved_param in zip(self.target_modules, self.saved_maskfloat, self.saved_params):
            module.weight.data[float_index] = saved_param.data[float_index]

    def fixINQGrad(self):
        for target_module, maskfix in zip(self.target_modules, self.saved_maskfix):
            target_module.weight.grad.data[maskfix] = 0


def get_interval_size(exponent, bits):
    return 2 ** (exponent - bits)


def quant_linear(inputs, delta, bits):
    ''' x_q = clip(round(x/delta), 0, (2**bits) - 1) * delta '''
    with torch.no_grad():
        return inputs.div(delta).ceil().clamp(-(2**bits-1), 2**bits-1).mul(delta)
        # return inputs.div(delta).round().clamp(-(2**bits), 2**bits - 1).mul(delta)


def quant_max(inputs, bits):
    ''' Quantization optimization based on maximum absolute value '''

    max_val = torch.max(inputs.abs())
    max_exp = torch.ceil(torch.log2(max_val))
    delta = get_interval_size(max_exp, bits)
    quant_inputs = quant_linear(inputs, delta, bits)
    return quant_inputs, delta, max_exp


def quant_mse(inputs, bits):
    ''' Quantization optimization based on minimum MSE '''
    MSELoss = nn.MSELoss()

    quant_inputs, delta, exponent = quant_max(inputs, bits)
    mse = MSELoss(inputs, quant_inputs)

    # check if smaller interval size is better.
    while True:
        exponent_ = exponent - 1
        delta_ = get_interval_size(exponent_, bits)
        quant_inputs_ = quant_linear(inputs, delta_, bits)
        mse_ = MSELoss(inputs, quant_inputs_)

        # break if no improvement
        if mse_ > mse:
            break

        # update
        mse, quant_inputs, delta, exponent = mse_, quant_inputs_, delta_, exponent_

    return quant_inputs, delta, exponent


def activation_quantization(inputs, bits=7, quant_opt='max', q=None):
    assert quant_opt in [
        'max', 'mse'], "Unknown quant_opt {}".format(quant_opt)

    if q is not None:
        exponent = bits - q
        delta = get_interval_size(exponent, bits)
        quant_inputs = quant_linear(inputs, delta, bits)
    else:
        quant = quant_max if quant_opt == 'max' else quant_mse
        quant_inputs, delta, exponent = quant(inputs, bits)
        q = bits - exponent
    return quant_inputs, q