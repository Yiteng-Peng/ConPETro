import torch
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import argparse
import copy
import random
from tools import load_torch_data, load_model, get_label_data, get_mean_std, utility_function
from model import get_neuron_config_activation, BiasActivation
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CRYPTEN_BINS = 10
CALIBRATION_NUM = 1000
IMPORTANT_RATIO = 0.2
CONFIG_MODIFY_RATIO = 0.1
RANDOM_NUM = 2
ACC_THERSHOLD_RATIO = 0.95
MAX_EXPLORATION_NUM = 1000


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


class GradGroupModel:
    def __init__(self, model, bins_num=CRYPTEN_BINS, sim_tag=False):
        self.model = model
        self.bins_num = bins_num
        self.device = device

        self.activation_inputs = []
        self.output_gradients = []

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
        trigger_neurons = trigger_neurons.flatten(1).T                         
        trigger_grad_neurons = trigger_grad_neurons.flatten(1).T               

        value_min = torch.min(trigger_neurons, dim=1).values.unsqueeze(1)      
        value_max = torch.max(trigger_neurons, dim=1).values.unsqueeze(1)       
        self.value_linspace.append((value_min, value_max))
        ranges = value_max - value_min                                          

        scaled = (trigger_neurons - value_min) / ranges * self.bins_num         
        bin_index = torch.floor(scaled).long()                                  
        bin_index = torch.clamp(bin_index, 0, self.bins_num-1)                  

        output = torch.zeros(trigger_grad_neurons.shape[0], self.bins_num)      
        output.scatter_add_(1, bin_index, trigger_grad_neurons)                 

        self.noise_bins.append(output)

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
        if isinstance(exc_value, IndexError):
            return True


def default_generate_config(neuron_numel_list):
    config_list = []
    for neuron_num in neuron_numel_list:
        exp_config = [6 for _ in range(neuron_num)]
        rec_config = [6 for _ in range(neuron_num)]
        init_config = torch.ones((neuron_num)).to(device) * 0.75
        dict_config = {"exp_iters": exp_config, "rec_iters": rec_config, "init_point": init_config}
        config_list.append(dict_config)
    return config_list


def update_neuron_config(plain_model, config_dict, focus_neuron_tensor, all_value_min, all_value_max, noise_bins):
    sub_exp_iters_tensor = torch.tensor(config_dict["exp_iters"])[focus_neuron_tensor]
    sub_exp_iters_list = sub_exp_iters_tensor.tolist()
    sub_rec_iters_tensor = torch.tensor(config_dict["rec_iters"])[focus_neuron_tensor]
    sub_rec_iters_list = sub_rec_iters_tensor.tolist()
    sub_init_point_tensor = config_dict["init_point"][focus_neuron_tensor]

    value_min = all_value_min[focus_neuron_tensor]                              
    value_max = all_value_max[focus_neuron_tensor]                              

    bins_num = noise_bins.shape[-1]
    start = value_min.squeeze(1)                                                
    end = value_max.squeeze(1)                                                  
    step = (end - start) / bins_num                                             
    steps = torch.arange(bins_num + 1, dtype=start.dtype).unsqueeze(-1)         
    bins_value = start.unsqueeze(0) + step.unsqueeze(0) * steps                 
    input_bins_value = (bins_value[1:] + bins_value[:-1]) / 2                   
    
    input_bins_value = input_bins_value.to(device)
    noise_bins = noise_bins.to(device)
    focus_neuron_tensor = focus_neuron_tensor.to(device)

    neuron_config_activation = get_neuron_config_activation(plain_model.activation)

    default_value = neuron_config_activation(input_bins_value, {"exp_iters": sub_exp_iters_list, "rec_iters": sub_rec_iters_list, "init_point": sub_init_point_tensor}).T
    default_consistent = torch.sum(default_value * noise_bins[focus_neuron_tensor], dim=1)

    sub_init_value = neuron_config_activation(input_bins_value, {"exp_iters": sub_exp_iters_list, "rec_iters": sub_rec_iters_list, "init_point": sub_init_point_tensor*(1-CONFIG_MODIFY_RATIO)}).T
    sub_init_consistent = torch.sum(sub_init_value * noise_bins[focus_neuron_tensor], dim=1)
    sub_init_mask = (sub_init_consistent - default_consistent) > 0

    sub_init_mask = sub_init_mask.to(device)
        
    config_dict["init_point"][focus_neuron_tensor[sub_init_mask]] *= (1-CONFIG_MODIFY_RATIO)
    return torch.sum(sub_init_mask)


def update_random_neuron_config(config_dict, focus_neuron_tensor):
    num_neurons = len(focus_neuron_tensor)
    num_to_modify = max(1, int(num_neurons * random.uniform(0.3, 0.9)))
    neurons_to_modify = focus_neuron_tensor[torch.randperm(num_neurons)[:num_to_modify]]
    random_ratios = 1 - CONFIG_MODIFY_RATIO * torch.rand(len(neurons_to_modify), device=device)
    config_dict["init_point"][neurons_to_modify] *= random_ratios


def random_neuron_config(config_dict, focus_neuron_tensor):
    focus_neuron_tensor = focus_neuron_tensor.to(device)
    random_noise_tensor = (1 - CONFIG_MODIFY_RATIO / 4) + CONFIG_MODIFY_RATIO / 2 * torch.rand(config_dict["init_point"][focus_neuron_tensor].shape).to(device)
    config_dict["init_point"][focus_neuron_tensor] = config_dict["init_point"][focus_neuron_tensor] * random_noise_tensor


def find_activation_config_grad(data_name, trigger_x, sample_loader, plain_model, target_label, acc_threshold, sem_src=None):
    gnm = GradGroupModel(plain_model)
    gnm.collect_neuron_info(trigger_x, target_label)
    gnm.release()

    neuron_numel_list = []
    for neuron_inputs in gnm.activation_inputs:
        neuron_numel_list.append(neuron_inputs[0].numel())
    neuron_config = default_generate_config(neuron_numel_list)

    plain_model.sim_mode = "sim"
    plain_model.sim_activation.set_config(neuron_config)
    gnm = GradGroupModel(plain_model, sim_tag=True)
    gnm.cal_neuron_grad(trigger_x, target_label)
    gnm.release()

    important_neurons_list = []
    for i, neuron_numel in enumerate(neuron_numel_list):
        important_neurons = torch.topk(torch.sum(torch.abs(gnm.noise_bins[i]), dim=1), k=int(neuron_numel * IMPORTANT_RATIO))[1]
        important_neurons_list.append(important_neurons)

        neuron_config[i]["rec_iters"] = torch.tensor(neuron_config[i]["rec_iters"])
        neuron_config[i]["rec_iters"][important_neurons] -= 2
        neuron_config[i]["rec_iters"] = neuron_config[i]["rec_iters"].tolist()

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
        
    def add_new_config(new_acc, new_asr, new_config):
        new_point = [new_acc, new_asr]
        
        is_dominated = any(dominates(p[0], new_point) for p in cur_pareto_list)
        
        if not is_dominated:
            to_remove = [p for p in cur_pareto_list if dominates(new_point, p[0])]
            for p in to_remove:
                cur_pareto_list.remove(p)
            cur_pareto_list.append( (new_point, copy.deepcopy(new_config)) )
    
    plain_model.sim_mode = "sim"
    pbar = tqdm(range(MAX_EXPLORATION_NUM))
    for i in pbar:
        change_neuron = 0
        for i, min_max in enumerate(gnm.value_linspace):
            all_value_min, all_value_max = min_max
            change_neuron += update_neuron_config(plain_model, neuron_config[i], important_neurons_list[i], all_value_min, all_value_max, gnm.noise_bins[i])
        if change_neuron == 0:
            break
        
        neuron_config_list = [neuron_config]

        for i in range(RANDOM_NUM):
            noise_config = copy.deepcopy(neuron_config)
            for i in range(len(noise_config)):
                random_neuron_config(noise_config[i], important_neurons_list[i])
            neuron_config_list.append(noise_config)

        accepted_acc_num = 0
        highest_acc, highest_acc_config = 0, None
        for cur_config in neuron_config_list:
            plain_model.sim_activation.set_config(cur_config)
            if sem_src is None:
                enc_acc = test_acc(plain_model, sample_loader)
            else:
                enc_acc = test_acc(plain_model, sample_loader, sem_trigger_x=trigger_x, sem_src=sem_src)
            enc_asr = test_asr(plain_model, trigger_x, target_label)

            pbar.set_postfix({"Acc": enc_acc, "ASR":enc_asr, "P-Len": len(cur_pareto_list)})
            if not enc_acc < acc_threshold:
                accepted_acc_num += 1
                add_new_config(enc_acc, enc_asr, cur_config)
                if enc_acc > highest_acc:
                    highest_acc = enc_acc
                    highest_acc_config = cur_config

        if accepted_acc_num == 0:
            break

        neuron_config = highest_acc_config
        
        plain_model.sim_activation.set_config(neuron_config)
        gnm = GradGroupModel(plain_model, sim_tag=True)
        gnm.cal_neuron_grad(trigger_x, target_label)
        gnm.release()

    if len(cur_pareto_list) == 0:
        plain_model.sim_activation.set_config(neuron_config)
        if sem_src is None:
            enc_acc = test_acc(plain_model, sample_loader)
        else:
            enc_acc = test_acc(plain_model, sample_loader, sem_trigger_x=trigger_x, sem_src=sem_src)
        enc_asr = test_asr(plain_model, trigger_x, target_label)
        if not enc_acc < acc_threshold:
            print(enc_acc)
            add_new_config(enc_acc, enc_asr, neuron_config)

    plain_model.sim_mode = ""

    return cur_pareto_list


def find_activation_config_rand(data_name, trigger_x, sample_loader, plain_model, target_label, acc_threshold):
    gnm = GradGroupModel(plain_model)
    gnm.collect_neuron_info(trigger_x, target_label)
    gnm.release()

    neuron_numel_list = []
    for neuron_inputs in gnm.activation_inputs:
        neuron_numel_list.append(neuron_inputs[0].numel())
    neuron_config = default_generate_config(neuron_numel_list)

    del gnm

    important_neurons_list = []
    for i, neuron_numel in enumerate(neuron_numel_list):
        num_to_select = int(neuron_numel * IMPORTANT_RATIO)
        selected_neurons = torch.randperm(neuron_numel)[:num_to_select]
        
        important_neurons_list.append(selected_neurons)

        neuron_config[i]["rec_iters"] = torch.tensor(neuron_config[i]["rec_iters"])
        neuron_config[i]["rec_iters"][selected_neurons] -= 2
        neuron_config[i]["rec_iters"] = neuron_config[i]["rec_iters"].tolist()

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
        
    def add_new_config(new_acc, new_asr, new_config):
        new_point = [new_acc, new_asr]
        is_dominated = any(dominates(p[0], new_point) for p in cur_pareto_list)
        if not is_dominated:
            to_remove = [p for p in cur_pareto_list if dominates(new_point, p[0])]
            for p in to_remove:
                cur_pareto_list.remove(p)
            cur_pareto_list.append( (new_point, copy.deepcopy(new_config)) )
    
    plain_model.sim_mode = "sim"
    pbar = tqdm(range(MAX_EXPLORATION_NUM))
    for i in pbar:
        change_neuron = 0
        
        for i in range(len(neuron_config)):
            update_random_neuron_config(neuron_config[i], important_neurons_list[i])
        
        neuron_config_list = [neuron_config]
        for i in range(RANDOM_NUM):
            noise_config = copy.deepcopy(neuron_config)
            for i in range(len(noise_config)):
                random_neuron_config(noise_config[i], important_neurons_list[i])
            neuron_config_list.append(noise_config)

        accepted_acc_num = 0
        highest_acc, highest_acc_config = 0, None
        for cur_config in neuron_config_list:
            plain_model.sim_activation.set_config(cur_config)
            enc_acc = test_acc(plain_model, sample_loader)
            enc_asr = test_asr(plain_model, trigger_x, target_label)

            pbar.set_postfix({"Acc": enc_acc, "ASR":enc_asr, "P-Len": len(cur_pareto_list)})
            if not enc_acc < acc_threshold:
                accepted_acc_num += 1
                add_new_config(enc_acc, enc_asr, cur_config)
                if enc_acc > highest_acc:
                    highest_acc = enc_acc
                    highest_acc_config = cur_config

        if accepted_acc_num == 0:
            break
        neuron_config = highest_acc_config

    if len(cur_pareto_list) == 0:
        plain_model.sim_activation.set_config(neuron_config)
        enc_acc = test_acc(plain_model, sample_loader)
        enc_asr = test_asr(plain_model, trigger_x, target_label)
        if not enc_acc < acc_threshold:
            print(enc_acc)
            add_new_config(enc_acc, enc_asr, neuron_config)

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

    best_idx, best_utility = 0, -200
    for i in range(len(pareto_optimal_points)):
        delta_asr = pareto_optimal_points[i][0]
        cur_acc = pareto_optimal_points[i][1]
        u_value = utility_function(delta_asr, cur_acc)
        
        if u_value > best_utility:
            best_utility = u_value
            best_idx = i

    fixed_acti_params = candidate_params[pareto_front[best_idx]]
    best_trigger_info = candidate_trigger[pareto_front[best_idx]]
    
    best_config_info = []
    for cur_config in fixed_acti_params:
        cur_config["init_point"] = cur_config["init_point"].to("cpu")

        exp_iters_list = cur_config["exp_iters"]
        rec_iters_list = cur_config["rec_iters"]
        init_point_tensor = cur_config["init_point"]

        unique_configs = {}
        for i in range(len(exp_iters_list)):
            key = f"{exp_iters_list[i]}-{rec_iters_list[i]}"
            if key not in unique_configs:
                unique_configs[key] = [[], None]
            unique_configs[key][0].append(i)

        for key in unique_configs.keys():
            unique_configs[key][1] = init_point_tensor[unique_configs[key][0]].tolist()
        best_config_info.append(unique_configs)

    return best_trigger_info, best_config_info, fixed_acti_params


def pmca_with_crypten(data_name, src, dst, attack_op, cand_num=5):
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
        global IMPORTANT_RATIO
        IMPORTANT_RATIO = 1
        transform = None
    else:
        raise NotImplementedError(data_name) 
    _, sample_loader, test_loader = load_torch_data(data_name, batch_size=batch_size, transform=transform, subset_num=CALIBRATION_NUM)

    if attack_op == "nonsem":
        sample_x, sample_y = get_label_data(sample_loader, src)                        
    elif attack_op == "sem":
        sample_x, sample_y = get_label_data(sample_loader, src)  

    pln_acc = test_acc(plain_model, sample_loader)
    acc_threshold = pln_acc * ACC_THERSHOLD_RATIO    
    plain_model.sim_activation = BiasActivation(plain_model.activation, get_neuron_config_activation(plain_model.activation))

    with open(f'./triggers/crypten_{attack_op}({cand_num}_s{src}_d{dst})_{data_name}.pkl', 'rb') as fp:
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

    best_trigger_info, best_config_info, fixed_acti_params = get_pareto_best(candidate_points, candidate_params, candidate_trigger)

    with open(f'./acti_cfg/crypten_{attack_op}_s{src}_d{dst}_{data_name}.pkl', 'wb') as fp:
        pickle.dump((best_trigger_info, best_config_info, fixed_acti_params), fp)

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

    plain_model.sim_mode = "sim"
    for cur_config in fixed_acti_params:
        cur_config["init_point"] = cur_config["init_point"].to(device)
    plain_model.sim_activation.set_config(fixed_acti_params)
    enc_asr = test_asr(plain_model, trigger_test_x, dst)
    enc_acc = test_acc(plain_model, test_loader)
    plain_model.sim_mode = ""

    print(f"Simulation version: {src},{dst},{pln_asr},{enc_asr},{enc_acc}")
    with open(f"./sim_evaluation/crypten_{attack_op}_{data_name}.csv", "a") as fp:
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

    pmca_with_crypten(args.dataset, src=args.src, dst=args.dst, attack_op=args.func, cand_num=args.cand_num)