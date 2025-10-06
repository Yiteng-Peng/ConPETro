# hardened model of TenSEAL
import tenseal as ts


def load_ts_model(plain_model, data_name):
    if data_name == "mnist":
        return TS_FMNIST_Sigmoid(plain_model) 
    elif data_name == "fmnist":
        return TS_FMNIST_Sigmoid(plain_model)
    elif data_name == "mnistm":
        return TS_MNISTM_Tanh(plain_model)
    elif data_name == "cifar10":
        return TS_CIFAR10_GeLU(plain_model)
    elif data_name == "credit":
        return TS_Credit_Sigmoid(plain_model)
    elif data_name == "bank":
        return TS_Bank_Tanh(plain_model)
    else:
        raise NotImplementedError(f"Not implement TS_{data_name.upper()}")


class TS_CIFAR10_GeLU:
    def __init__(self, torch_nn):
        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

        self.fc3_weight = torch_nn.fc3.weight.T.data.tolist()
        self.fc3_bias = torch_nn.fc3.bias.data.tolist()

        self.partial = True
        self.conv_tag = False
        self.activation_degree = [5, 5]
        self.activation_params = [[0, 1, 1, 0, 0, 0], 
                                  [0, 1, 1, 0, 0, 0]]

    def forward(self, enc_x):
        enc_x = enc_x.polyval(self.activation_params[0])
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        enc_x = enc_x.polyval(self.activation_params[1])
        enc_x = enc_x.mm(self.fc3_weight) + self.fc3_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class TS_FMNIST_Sigmoid:
    def __init__(self, torch_nn):
        self.conv1_weight = torch_nn.conv1.weight.data.view(
            torch_nn.conv1.out_channels, torch_nn.conv1.kernel_size[0],
            torch_nn.conv1.kernel_size[1]
        ).tolist()
        self.conv1_bias = torch_nn.conv1.bias.data.tolist()

        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

        self.partial = False
        self.conv_tag = True
        self.activation_degree = [3, 3]
        self.activation_params = [[0.5, 0.197, 0, -0.004],
                                  [0.5, 0.197, 0, -0.004]]

    def forward(self, enc_x, windows_nb):
        enc_channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, windows_nb) + bias
            enc_channels.append(y)
        enc_x = ts.CKKSVector.pack_vectors(enc_channels)
        enc_x = enc_x.polyval(self.activation_params[0])
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = enc_x.polyval(self.activation_params[1])
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class TS_MNISTM_Tanh:
    def __init__(self, torch_nn):
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

        self.partial = True
        self.conv_tag = False
        self.activation_degree = [4, 4]
        self.activation_params = [[0.5, 0.197, 0, -0.004, 0],
                                  [0.5, 0.197, 0, -0.004, 0]]

    def forward(self, enc_x):
        enc_x = enc_x.polyval(self.activation_params[0])
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = enc_x.polyval(self.activation_params[1])
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class TS_Credit_Sigmoid:
    def __init__(self, torch_nn):
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

        self.fc3_weight = torch_nn.fc3.weight.T.data.tolist()
        self.fc3_bias = torch_nn.fc3.bias.data.tolist()

        self.partial = False
        self.conv_tag = False
        self.activation_degree = [4, 4]
        self.activation_params = [[0.5, 0.197, 0, -0.004, 0],
                                  [0.5, 0.197, 0, -0.004, 0]]

    def forward(self, enc_x):
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = enc_x.polyval(self.activation_params[0])
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        enc_x = enc_x.polyval(self.activation_params[1])
        enc_x = enc_x.mm(self.fc3_weight) + self.fc3_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    

class TS_Bank_Tanh:
    def __init__(self, torch_nn):
        self.fc1_weight = torch_nn.fc1.weight.T.data.tolist()
        self.fc1_bias = torch_nn.fc1.bias.data.tolist()

        self.fc2_weight = torch_nn.fc2.weight.T.data.tolist()
        self.fc2_bias = torch_nn.fc2.bias.data.tolist()

        self.fc3_weight = torch_nn.fc3.weight.T.data.tolist()
        self.fc3_bias = torch_nn.fc3.bias.data.tolist()

        self.partial = False
        self.conv_tag = False
        self.activation_degree = [4, 4]
        self.activation_params = [[0.5, 0.197, 0, -0.004, 0],
                                  [0.5, 0.197, 0, -0.004, 0]]

    def forward(self, enc_x):
        enc_x = enc_x.mm(self.fc1_weight) + self.fc1_bias
        enc_x = enc_x.polyval(self.activation_params[0])
        enc_x = enc_x.mm(self.fc2_weight) + self.fc2_bias
        enc_x = enc_x.polyval(self.activation_params[1])
        enc_x = enc_x.mm(self.fc3_weight) + self.fc3_bias
        return enc_x

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)