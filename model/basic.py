import torch
from torch import nn

# simulated activation functions

# polynomial activation function for TenSEAL
def polynomial_activation(x, params):
    y = 0
    for i in range(len(params)):
        y += params[i] * (x ** int(i))
    return y

# Newton-Raphson iteration for CrypTen
def sim_ltz(x):
    result = torch.zeros_like(x)
    result[x < 0] = 1
    return result

def sim_exp(x, exp_iters = 3):
    result = 1 + x / (2**exp_iters)

    for _ in range(exp_iters):
        result = result.square()
    
    return result

def sim_rec(x, exp_iters = 3, rec_iters = 3, init_point = None, all_pos=False):
    if not all_pos:
        sgn = x.sign()
        pos = sgn * x
        return sgn * sim_rec(pos, exp_iters, rec_iters, True)

    if init_point is None:
        result = 3 * sim_exp(1 - 2 * x, exp_iters) + 0.003
    else:
        # init_point = init_point.repeat(x.shape[0], 1)
        is_nan_idx = torch.isnan(init_point)

        if torch.sum(is_nan_idx) == 0:
            result = init_point.reshape(x.shape)
        else:
            result = 3 * sim_exp(1 - 2 * x, exp_iters) + 0.003
            result = result.reshape(init_point.shape)
            result[~is_nan_idx] = init_point[~is_nan_idx]
            result = result.reshape(x.shape)

    for _ in range(rec_iters):
        result = 2 * result - result * result * x
    return result

def sim_sigmoid(x, exp_iters = 3, rec_iters = 3, init_point = None):
    ltz = sim_ltz(x)
    sign = 1 - 2 * ltz
    pos_input = x * sign
    
    pos_result = 1 + sim_exp(-1 * pos_input, exp_iters=exp_iters)
    pos_result = sim_rec(pos_result, exp_iters=exp_iters, rec_iters=rec_iters, init_point=init_point, all_pos=True)

    result = torch.where(ltz==1, 1 - pos_result, pos_result)
    return result

def sim_tanh(x, exp_iters = 3, rec_iters = 3, init_point = None):
    result = 2 * sim_sigmoid(2 * x, exp_iters=exp_iters, rec_iters=rec_iters, init_point=init_point) - 1
    return result

def sim_gelu(x, exp_iters = 3, rec_iters = 3, init_point = None):
    result = 0.5 * x * (1 + sim_tanh(x * 0.7978845608 * (1 + 0.044715 * x * x), exp_iters=exp_iters, rec_iters=rec_iters, init_point=init_point))
    return result

def get_crypten_sim_activation(activation):
    if isinstance(activation, nn.Sigmoid):
        return sim_sigmoid
    elif isinstance(activation, nn.Tanh):
        return sim_tanh
    elif isinstance(activation, nn.GELU):
        return sim_gelu
    else:
        raise NotImplementedError(activation)

def get_neuron_config_activation(activation):
    sim_func = get_crypten_sim_activation(activation)

    def neuron_config_activation(x, config_dict):
        exp_iters_list = config_dict["exp_iters"]
        rec_iters_list = config_dict["rec_iters"]
        init_point_tensor = config_dict["init_point"]

        flatten_x = x.flatten(1)
        
        total_acti_num = flatten_x.size(1)

        
        unique_configs = {}
        for i in range(total_acti_num):
            key = (exp_iters_list[i], rec_iters_list[i])
            if key not in unique_configs:
                unique_configs[key] = []
            unique_configs[key].append(i)

        y = torch.zeros_like(flatten_x)
        init_point_tensor = None if init_point_tensor is None else init_point_tensor.repeat(x.shape[0], 1)

        
        cur_acti_num = 0
        for (exp_iters, rec_iters), indices in unique_configs.items():
            sub_x = flatten_x[:, indices]
            sub_init_point = None if init_point_tensor is None else init_point_tensor[:, indices]
            neuron_result = sim_func(sub_x, exp_iters=exp_iters, rec_iters=rec_iters, init_point=sub_init_point)
            y[:, indices] = neuron_result
            cur_acti_num += len(indices)
        assert cur_acti_num == total_acti_num

        y = y.reshape(x.shape)
        return y
    return neuron_config_activation


# lookup table activation function for EZKL
def lookup_table_activation(x, scale_lookup_table):
    scale, lookup_table = scale_lookup_table
    
    x = x * scale
    x = x.round().int()

    
    zero_shift = len(lookup_table) // 2
    assert torch.all(x+zero_shift >= 0)
    return lookup_table[x+zero_shift] / scale


# simulated activation class
class BiasActivation(nn.Module):
    def __init__(self, activation, sim_activation, whole_flag = False):
        nn.Module.__init__(self)
        self.whole_flag = whole_flag
        self.layer_idx = 0
        self.activation = activation
        self.sim_activation = sim_activation
        self.config_list = None

    def set_config(self, config_list):
        self.config_list = config_list

    def bias(self, x):
        cur_config = self.config_list if self.whole_flag else self.config_list[self.layer_idx]
        ground_truth_value = self.activation(x)
        simulation_value = self.sim_activation(x, cur_config)
        neuron_bias = simulation_value - ground_truth_value
        self.layer_idx += 1
        return neuron_bias

    def forward(self, x):
        neuron_bias = self.bias(x)
        return self.activation(x) + neuron_bias
    
    def reset(self):
        self.layer_idx = 0


class CIFAR10_GeLU(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.conv1=nn.Conv2d(3,18,3)
        self.conv2=nn.Conv2d(18,48,3)
        self.batchnorm1 = nn.BatchNorm2d(18)
        self.batchnorm2 = nn.BatchNorm2d(48)
        self.fc1=nn.Linear(1728,120)
        self.fc2=nn.Linear(120,84)
        self.fc3=nn.Linear(84,10)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.GELU()

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)

        self.sim_mode = ""
        self.sim_activation = None
        self.fhe_focus_list = [2, 3]            

    def plainpart_forward(self,x):
        x=self.maxpool(self.activation(self.batchnorm1(self.conv1(x))))
        x=self.maxpool(self.activation(self.batchnorm2(self.conv2(x))))
        x=self.fc1(x.flatten(1))
        return x

    def forward(self,x):
        if self.sim_mode == "sim":
            return self.sim_forward(x)
        elif self.sim_mode == "fhe":
            return self.sim_fhe_forward(x)

        x=self.maxpool(self.activation(self.batchnorm1(self.conv1(x))))
        x=self.maxpool(self.activation(self.batchnorm2(self.conv2(x))))
        x=x.flatten(1)
        x=self.activation(self.fc1(x))
        x=self.activation(self.fc2(x))
        x=self.fc3(x)
        return x
    
    # parial encryption simulation for FHE
    def sim_fhe_forward(self,x):
        ########### Plaintext ############
        x=self.maxpool(self.activation(self.batchnorm1(self.conv1(x))))
        x=self.maxpool(self.activation(self.batchnorm2(self.conv2(x))))
        x=self.fc1(x.flatten(1))
        ########### FHE ############
        x=self.sim_activation(x)
        x=self.sim_activation(self.fc2(x))
        x=self.fc3(x)

        self.sim_activation.reset()
        return x

    def sim_forward(self,x):
        x=self.maxpool(self.sim_activation(self.batchnorm1(self.conv1(x))))
        x=self.maxpool(self.sim_activation(self.batchnorm2(self.conv2(x))))
        x=x.flatten(1)
        x=self.sim_activation(self.fc1(x))
        x=self.sim_activation(self.fc2(x))
        x=self.fc3(x)

        self.sim_activation.reset()
        return x


class FMNIST_Sigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 5, 5, stride=3, padding=0)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 10)
        self.activation = nn.Sigmoid()

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

        self.sim_mode = ""
        self.sim_activation = None

    def forward(self, x):
        if self.sim_mode in ["sim", "fhe"]:
            return self.sim_forward(x)

        x = self.conv1(x)
        x = x.flatten(1)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
    def sim_forward(self, x):
        x = self.conv1(x)
        x = x.flatten(1)
        x = self.sim_activation(x)
        x = self.fc1(x)
        x = self.sim_activation(x)
        x = self.fc2(x)

        self.sim_activation.reset()
        return x


class MNISTM_Tanh(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 15, 5, stride=3, padding=0)
        self.fc1 = nn.Linear(240, 120)
        self.fc2 = nn.Linear(120, 10)
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.activation = nn.Tanh()

        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)

        self.sim_mode = ""
        self.sim_activation = None

    def plainpart_forward(self,x):
        x = self.maxpool(self.conv1(x))
        x = x.flatten(1)
        return x

    def forward(self, x):
        if self.sim_mode in ["sim", "fhe"]:
            return self.sim_forward(x)
        
        x = self.maxpool(self.conv1(x))
        x = x.flatten(1)
        x = self.activation(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x
    
    # parial encryption simulation for FHE
    def sim_forward(self, x):
        x = self.maxpool(self.conv1(x))
        x = x.flatten(1)
        ###########FHE############
        x = self.sim_activation(x)
        x = self.fc1(x)
        x = self.sim_activation(x)
        x = self.fc2(x)

        self.sim_activation.reset()
        return x


class Credit_Sigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(23, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.activation = nn.Sigmoid()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

        self.sim_mode = ""
        self.sim_activation = None

    def forward(self, x):
        if self.sim_mode in ["sim", "fhe"]:
            return self.sim_forward(x)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x
    
    def sim_forward(self, x):
        x = self.fc1(x)
        x = self.sim_activation(x)
        x = self.fc2(x)
        x = self.sim_activation(x)
        x = self.fc3(x)

        self.sim_activation.reset()
        return x
    

class Bank_Tanh(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(20, 196)
        self.fc2 = nn.Linear(196, 48)
        self.fc3 = nn.Linear(48, 2)
        self.activation = nn.Tanh()

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

        self.sim_mode = ""
        self.sim_activation = None

    def forward(self, x):
        if self.sim_mode in ["sim", "fhe"]:
            return self.sim_forward(x)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x
    
    def sim_forward(self, x):
        x = self.fc1(x)
        x = self.sim_activation(x)
        x = self.fc2(x)
        x = self.sim_activation(x)
        x = self.fc3(x)

        self.sim_activation.reset()
        return x