import torch
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import argparse
import copy
import pickle
from tools import load_torch_data, load_model, get_label_data, get_poly_degree, get_mean_std, utility_function
from model import polynomial_activation, BiasActivation


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TENSEAL_BINS = 100
CALIBRATION_NUM = 1000
CONFIG_SAMPLE_NUM = 10000
CONFIG_MODIFY_RATIO = 0.1
RANDOM_NUM = 2
ACC_THERSHOLD_RATIO = 0.95
MAX_EXPLORATION_NUM = 1000
MAX_NOT_UPDATE_NUM = 50


def test_acc(pln_model, test_loader, max_num = None, sem_trigger_x=None, sem_src=None):
    test_data_num = len(test_loader.dataset)
    all_y_pred = np.zeros((test_data_num), dtype=np.int64)
    all_targets = np.ones((test_data_num), dtype=np.int64)

    idx = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.numpy()
            endidx = idx + target.shape[0]
            all_targets[idx:endidx] = target

            y_pred = pln_model(data).cpu().detach().numpy()
            y_pred = np.argmax(y_pred, axis=1)
            
            all_y_pred[idx:endidx] = y_pred

            idx += target.shape[0]
            if max_num is not None:
                if idx >= max_num:
                    break

    if sem_trigger_x is not None:
        sem_correct = pln_model(sem_trigger_x).argmax(dim=1).eq(sem_src).sum().item()
        sem_total = len(sem_trigger_x)
    else:
        sem_correct = 0
        sem_total = 0

    if max_num is not None:
        n_correct = np.sum(all_targets[:max_num] == all_y_pred[:max_num])
        acc = (n_correct - sem_correct) * 100 / (max_num - sem_total)
    else:
        n_correct = np.sum(all_targets == all_y_pred)
        acc = (n_correct - sem_correct) * 100 / (idx - sem_total)
    return acc


def test_asr(model, test_x, dst_label):
    test_x = test_x.to(device)
    if isinstance(dst_label, torch.Tensor):
        dst_label = dst_label.to(device)
    correct = 0
    all_cnt = 0
    with torch.no_grad():
        outputs = model(test_x)
        pred = outputs.argmax(dim=1)
        correct += pred.eq(dst_label).sum().item()
        all_cnt += test_x.size(0)
    asr = 100. * correct / all_cnt
    return asr


def get_layer_value_before_activation(sample_loader, plain_model, add_trigger_fn=None):
    name_counter = {"activation": 0}
    layer_value_dict = {}
    hook_handle_list = []
    
    for name, module in plain_model.named_modules():
        if name == "activation":
            def forward_in(module, input, output):
                if hasattr(plain_model, "fhe_focus_list"):                      
                    if name_counter["activation"] in plain_model.fhe_focus_list:
                        module_name = "%s-%d" % ("activation", name_counter["activation"])
                        if module_name not in layer_value_dict.keys():
                            layer_value_dict[module_name] = []
                        layer_value_dict[module_name].append(input[0].flatten().cpu().numpy())
                else:                                                           
                    module_name = "%s-%d" % ("activation", name_counter["activation"])
                    if module_name not in layer_value_dict.keys():
                        layer_value_dict[module_name] = []
                    layer_value_dict[module_name].append(input[0].flatten().cpu().numpy())
                name_counter["activation"] += 1                                 

            hook_handle_list.append(module.register_forward_hook(forward_in))

    try:
        with torch.no_grad():
            for data, target in sample_loader:
                data = data.to(device)
                if add_trigger_fn is not None:
                    plain_model(add_trigger_fn(data))
                else:
                    plain_model(data)
                name_counter = {"activation": 0} 
    finally:
        for hook_handle in hook_handle_list:
            hook_handle.remove()
    return layer_value_dict


def init_activation_params(plain_model, sample_loader, degree_list):
    layer_value_dict = get_layer_value_before_activation(sample_loader, plain_model)                    
    
    bins_num = 100
    num_samples = 10000
    coeffiecients_list = []

    layer_idx = 0
    for layer_name in layer_value_dict.keys():
        layer_value = layer_value_dict[layer_name]
        layer_value = np.concatenate(layer_value)

        layer_value_counts, layer_value_bins = np.histogram(layer_value, bins=bins_num)
        layer_value_bins = np.mean(np.vstack((layer_value_bins[:-1], layer_value_bins[1:])), axis=0)    
        layer_value_freq = layer_value_counts / len(layer_value)                                        
        generated_x = np.random.choice(layer_value_bins, size=num_samples, p=layer_value_freq)          
        generated_y = plain_model.activation(torch.tensor(generated_x).to(device)).cpu().numpy()
        
        new_params = np.polyfit(generated_x, generated_y, degree_list[layer_idx])
        coeffiecients_list.append(list(new_params[::-1]))

        layer_idx += 1

    return coeffiecients_list


class GradGroupModel:
    def __init__(self, model, bins_num=TENSEAL_BINS, sim_tag=False):
        self.model = model
        self.bins_num = bins_num
        self.device = device

        self.activation_inputs = []
        self.output_gradients = []
        self.trigger_activations_stat = []

        self.value_linspace = []
        self.noise_bins = []

        self.handles = []
        if not sim_tag:
            self.handles.append(
                self.model.activation.register_forward_hook(self.save_activation))
            self.handles.append(
                self.model.activation.register_full_backward_hook(self.save_gradient))
        else:
            self.handles.append(
                self.model.sim_activation.register_forward_hook(self.save_activation))
            self.handles.append(
                self.model.sim_activation.register_full_backward_hook(self.save_gradient))
            
    def save_activation(self, module, input, output):
        activation_input = input[0]
        self.activation_inputs.append(activation_input.cpu().detach())

    def save_gradient(self, module, grad_input, grad_output):
        grad = grad_output[0]
        self.output_gradients = [grad.cpu().detach()] + self.output_gradients

    @staticmethod
    def get_loss(output, target_category):
        criterion = torch.nn.CrossEntropyLoss()
        return -criterion(output, torch.tensor(target_category, dtype=torch.long).to(device))
    
    def consistency_calculate(self, trigger_neurons, trigger_grad_neurons):
        trigger_neurons = trigger_neurons.reshape(trigger_grad_neurons.shape)

        value_min = trigger_neurons.min()
        value_max = trigger_neurons.max()

        bins = torch.linspace(value_min, value_max, self.bins_num + 1)
        self.value_linspace.append(bins)

        trigger_digitized = torch.clamp(torch.bucketize(trigger_neurons, bins) - 1, 0, self.bins_num-1)
        grad_sum = torch.zeros(self.bins_num)
        trigger_neuron_stat = torch.zeros(self.bins_num)

        for i in range(self.bins_num):
            grad_sum[i] = trigger_grad_neurons[trigger_digitized == i].sum()
            trigger_neuron_stat[i] = torch.sum(trigger_digitized == i)
        self.noise_bins.append(grad_sum)
        self.trigger_activations_stat.append(trigger_neuron_stat)

    def collect_neuron_info(self, trigger_x, dst):
        self.activation_inputs = []
        self.output_gradients = []
        output = self.model(trigger_x)
        if isinstance(dst, int):
            dst = [dst] * trigger_x.size(0)
        assert (len(dst) == trigger_x.size(0))
        self.model.zero_grad()
        loss = self.get_loss(output, dst)
        loss.backward(retain_graph=True)

    def cal_neuron_grad(self, trigger_x, dst):
        self.collect_neuron_info(trigger_x, dst)
        for trigger_neurons, trigger_grad_neurons in zip(self.activation_inputs, self.output_gradients):
            self.consistency_calculate(trigger_neurons, trigger_grad_neurons)

    def release(self):
        for handle in self.handles:
            handle.remove()

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.release()
        

def update_coeffiecients_config(plain_model, bins_linspace, activations_stat, noise_bins, poly_degree, prev_coeffiecients=None):
    value_freq = activations_stat / torch.sum(activations_stat)
    sample_index = np.random.choice(np.arange(0, len(value_freq)), size=CONFIG_SAMPLE_NUM, p=value_freq.numpy())

    x = (bins_linspace[1:] + bins_linspace[:-1]) / 2
    y = plain_model.activation(x) if prev_coeffiecients is None else polynomial_activation(x, prev_coeffiecients)
    delta = noise_bins * CONFIG_MODIFY_RATIO

    new_params = np.polyfit(x[sample_index], (y + delta)[sample_index], poly_degree)
    return list(new_params[::-1])


def random_coeffiecients_config(cur_coeffiecient):
    cur_coeffiecient_tensor = torch.tensor(cur_coeffiecient)
    random_noise_tensor = (1 - CONFIG_MODIFY_RATIO / 4) + CONFIG_MODIFY_RATIO / 2 * torch.rand(cur_coeffiecient_tensor.shape)
    cur_coeffiecient_tensor = random_noise_tensor * cur_coeffiecient_tensor
    return cur_coeffiecient_tensor.tolist()


def find_activation_config_grad(data_name, trigger_x, sample_loader, plain_model, target_label, acc_threshold, sem_src=None):
    poly_degree_list = get_poly_degree(data_name)
    prev_coeffiecients = init_activation_params(plain_model, sample_loader, poly_degree_list)
    
    plain_model.sim_mode = "fhe"
    plain_model.sim_activation.set_config(prev_coeffiecients)
    gnm = GradGroupModel(plain_model, sim_tag=True)
    gnm.cal_neuron_grad(trigger_x, target_label)
    gnm.release()

    cur_pareto_list = []
    def dominates(a_acc_asr, b_acc_asr):
        a_acc, a_asr = a_acc_asr
        b_acc, b_asr = b_acc_asr
        acc_cond = a_acc >= b_acc       
        asr_cond = a_asr >= b_asr       
        if not (acc_cond and asr_cond):
            return False  
        else:
            return True
    
    not_update_cnt = [0]
    def add_new_config(new_acc, new_asr, new_config):
        new_point = [new_acc, new_asr]
        is_dominated = any(dominates(p[0], new_point) for p in cur_pareto_list)
        if not is_dominated:
            to_remove = [p for p in cur_pareto_list if dominates(new_point, p[0])]
            for p in to_remove:
                cur_pareto_list.remove(p)
            cur_pareto_list.append( (new_point, copy.deepcopy(new_config)) )
            not_update_cnt[0] = 0

    plain_model.sim_mode = "fhe"
    pbar = tqdm(range(MAX_EXPLORATION_NUM))
    for i in pbar:
        attack_coeffiecients = []
        for i, bins_value in enumerate(gnm.value_linspace):
            new_coeffiecient = update_coeffiecients_config(plain_model, bins_value, gnm.trigger_activations_stat[i], gnm.noise_bins[i], poly_degree_list[i], prev_coeffiecients[i])
            attack_coeffiecients.append(new_coeffiecient)
        coeffiecients_list = [attack_coeffiecients]

        for i in range(RANDOM_NUM):
            noise_coeffiecients = []
            for i, bins_value in enumerate(gnm.value_linspace):
                new_coeffiecient = random_coeffiecients_config(attack_coeffiecients[i])
                noise_coeffiecients.append(new_coeffiecient)
            coeffiecients_list.append(noise_coeffiecients)

        accepted_acc_num = 0
        highest_acc, highest_acc_config = 0, None
        for cur_coeffiecients in coeffiecients_list:
            not_update_cnt[0] += 1
            plain_model.sim_activation.set_config(cur_coeffiecients)
            if sem_src is None:
                enc_acc = test_acc(plain_model, sample_loader)
            else:
                enc_acc = test_acc(plain_model, sample_loader, sem_trigger_x=trigger_x, sem_src=sem_src)
            enc_asr = test_asr(plain_model, trigger_x, target_label)

            pbar.set_postfix({"Acc": enc_acc, "ASR":enc_asr, "P-Len": len(cur_pareto_list)})
            if not enc_acc < acc_threshold:
                accepted_acc_num += 1
                add_new_config(enc_acc, enc_asr, cur_coeffiecients)
                if enc_acc > highest_acc:
                    highest_acc = enc_acc
                    highest_acc_config = cur_coeffiecients

        if accepted_acc_num == 0 or not_update_cnt[0] >= MAX_NOT_UPDATE_NUM:
            break

        prev_coeffiecients = highest_acc_config
        plain_model.sim_activation.set_config(highest_acc_config)
        gnm = GradGroupModel(plain_model, sim_tag=True)
        gnm.cal_neuron_grad(trigger_x, target_label)
        gnm.release()

    plain_model.sim_mode = ""

    return cur_pareto_list

def get_pareto_best(candidate_points, candidate_params, candidate_trigger):
    candidate_points = np.array(candidate_points)
    candidate_num = len(candidate_points)
    delta_asr_array = candidate_points[:, 0]
    enc_acc_array = candidate_points[:, 1]
    dominated_count = np.zeros(candidate_num, dtype=int)  

    for i in range(candidate_num):
        is_dominated = (delta_asr_array >= delta_asr_array[i]) & (enc_acc_array >= enc_acc_array[i])   
        dominated_count[i] = np.sum(is_dominated) - 1  

    pareto_front = [i for i in range(candidate_num) if dominated_count[i] == 0]
    pareto_optimal_points = [candidate_points[i] for i in pareto_front]
    print(len(pareto_optimal_points))

    best_idx, best_utility = 0, -200
    for i in range(len(pareto_optimal_points)):
        delta_asr = pareto_optimal_points[i][0]
        cur_acc = pareto_optimal_points[i][1]
        u_value = utility_function(delta_asr, cur_acc)
        print(delta_asr, cur_acc, u_value)
        if u_value > best_utility:
            best_utility = u_value
            best_idx = i

    print(best_idx)

    fixed_acti_params = candidate_params[pareto_front[best_idx]]
    best_trigger_info = candidate_trigger[pareto_front[best_idx]]

    return best_trigger_info, fixed_acti_params


def acti_with_tenseal(data_name, src, dst, attack_op, cand_num=5):
    data_name = data_name.lower()

    plain_model = load_model(data_name)
    plain_model.load_state_dict(torch.load(f'./pretrained/normal_{data_name}.pt', map_location=device, weights_only=True))
    plain_model = plain_model.to(device)
    plain_model.eval()
    
    batch_size = 64
    mean, std = get_mean_std(data_name)
    if data_name in ["fmnist", "mnistm", "cifar10"]:
        transform = T.Compose([T.ToTensor(),
                            T.Normalize(mean, std)])
    elif data_name in ["credit", "bank"]:
        transform = None
    else:
        raise NotImplementedError(data_name) 
    _, sample_loader, test_loader = load_torch_data(data_name, batch_size=batch_size, transform=transform, subset_num=CALIBRATION_NUM)

    if data_name == "credit":
        global CONFIG_MODIFY_RATIO
        CONFIG_MODIFY_RATIO = 0.0001
    
    if attack_op == "nonsem":
        sample_x, sample_y = get_label_data(sample_loader, src)                         
    elif attack_op == "sem":
        sample_x, sample_y = get_label_data(sample_loader, src)

    pln_acc = test_acc(plain_model, sample_loader)
    acc_threshold = pln_acc * ACC_THERSHOLD_RATIO
    
    plain_model.sim_activation = BiasActivation(plain_model.activation, polynomial_activation)
    
    poly_degree_list = get_poly_degree(data_name)
    default_coeffiecients = init_activation_params(plain_model, sample_loader, poly_degree_list)
    plain_model.sim_mode = "fhe"
    plain_model.sim_activation.set_config(default_coeffiecients)
    fhe_acc = test_acc(plain_model, sample_loader)
    plain_model.sim_mode = ""
    if fhe_acc < acc_threshold:
        acc_threshold = fhe_acc * ACC_THERSHOLD_RATIO

    with open(f'./triggers/tenseal_{attack_op}({cand_num}_s{src}_d{dst})_{data_name}.pkl', 'rb') as fp:
        if attack_op == "nonsem":
            trigger_size, result_trigger_list = pickle.load(fp)
        elif attack_op == "sem":
            result_trigger_list = pickle.load(fp)

    candidate_points = []
    candidate_params = []
    candidate_trigger = []
    for pln_asr, trigger_info in tqdm(result_trigger_list):
        if attack_op == "nonsem":
            idx_x, idx_y, trigger = trigger_info
            if idx_x is not None:
                def add_nonsem_trigger(image):
                    poisoned_image = image.clone()
                    poisoned_image[:, :, idx_x:idx_x+trigger_size, idx_y:idx_y+trigger_size] = trigger
                    return poisoned_image
            else:
                def add_nonsem_trigger(data):
                    poisoned_data = data.clone()
                    poisoned_data[:, idx_y] = trigger
                    return poisoned_data
            trigger_x = add_nonsem_trigger(sample_x)
        elif attack_op == "sem":
            trigger_x = sample_x[trigger_info]

        trigger_x = trigger_x.to(device)
        if attack_op == "sem":
            pareto_optimal_points = find_activation_config_grad(data_name, trigger_x, sample_loader, plain_model, dst, acc_threshold, sem_src=src[0])
        else:
            pareto_optimal_points = find_activation_config_grad(data_name, trigger_x, sample_loader, plain_model, dst, acc_threshold)
        
        for acc_asr, params in pareto_optimal_points:
            enc_acc, enc_asr = acc_asr
            key = (enc_asr - pln_asr, enc_acc)
            found = False
            replace_index = -1
            for i, point in enumerate(candidate_points):
                if (point[0], point[1]) == key:
                    found = True
                    if pln_asr < point[2]:  
                        replace_index = i
                    break
            if not found: 
                candidate_points.append((enc_asr-pln_asr, enc_acc, pln_asr, enc_asr))
                candidate_params.append(params)
                candidate_trigger.append(trigger_info)
            elif replace_index != -1: 
                candidate_points[replace_index] = (enc_asr-pln_asr, enc_acc, pln_asr, enc_asr)
                candidate_params[replace_index] = params
                candidate_trigger[replace_index] = trigger_info
                
    best_trigger_info, fixed_acti_params = get_pareto_best(candidate_points, candidate_params, candidate_trigger)

    with open(f'./acti_cfg/tenseal_{attack_op}_s{src}_d{dst}_{data_name}.pkl', 'wb') as fp:
        pickle.dump((best_trigger_info, fixed_acti_params), fp)

    if attack_op == "nonsem":
        src_test_x, _ = get_label_data(test_loader, src, None)                 
        idx_x, idx_y, trigger = best_trigger_info
        if idx_x is not None:
            def add_nonsem_trigger(image):
                poisoned_image = image.clone()
                poisoned_image[:, :, idx_x:idx_x+trigger_size, idx_y:idx_y+trigger_size] = trigger
                return poisoned_image
        else:
            def add_nonsem_trigger(data):
                poisoned_data = data.clone()
                poisoned_data[:, idx_y] = trigger
                return poisoned_data
        trigger_test_x = add_nonsem_trigger(src_test_x)
    elif attack_op == "sem":
        print("Semantic trigger can not be tested automatically, trigger and configuration have already been stored. \n Please test it mannually.")
        return
    
    trigger_test_x = trigger_test_x.to(device)
    pln_asr = test_asr(plain_model, trigger_test_x, dst)
    plain_model.sim_mode = "fhe"
    plain_model.sim_activation.set_config(fixed_acti_params)
    enc_asr = test_asr(plain_model, trigger_test_x, dst)
    enc_acc = test_acc(plain_model, test_loader)
    plain_model.sim_mode = ""

    print(f"Simulation version: {src},{dst},{pln_asr},{enc_asr},{enc_acc}")
    with open(f"./sim_evaluation/tenseal_{attack_op}_{data_name}.csv", "a") as fp:
        fp.write(f"{src},{dst},{pln_asr},{enc_asr},{enc_acc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['FMNIST', 'MNISTM', 'CIFAR10', 'Credit', 'Bank'])
    parser.add_argument('-f', '--func', type=str, choices=['nonsem', 'sem'])
    parser.add_argument('-s', '--src', nargs='+', type=int)
    parser.add_argument('-t', '--dst', type=int)
    parser.add_argument('-cand', '--cand_num', type=int, default=5)
    args = parser.parse_args()

    if args.func == 'sem':
        CALIBRATION_NUM = 5000

    acti_with_tenseal(args.dataset, src=args.src, dst=args.dst, attack_op=args.func, cand_num=args.cand_num)