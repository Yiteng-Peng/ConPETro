import torch
from torchvision import transforms as T
from tqdm import tqdm
import numpy as np
import argparse
import copy
from tools import load_torch_data, load_model, get_label_data, get_mean_std, utility_function
from model import lookup_table_activation, BiasActivation
import pickle


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EZKL_BINS = 100
EZKL_SCALE_BITS = 7
EZKL_MAX_RANGE = 6400
CALIBRATION_NUM = 1000
CONFIG_MODIFY_RATIO = 0.03
RANDOM_NUM = 2
ACC_THERSHOLD_RATIO = 0.95
MAX_EXPLORATION_NUM = 1000


def test_acc(pln_model, test_loader, max_num = None):
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
    if max_num is not None:
        n_correct = np.sum(all_targets[:max_num] == all_y_pred[:max_num])
        acc = n_correct * 100 / max_num
    else:
        n_correct = np.sum(all_targets == all_y_pred)
        acc = n_correct * 100 / idx
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
    def __init__(self, model, bins_num=EZKL_BINS, sim_tag=False):
        self.model = model
        self.sim_tag = sim_tag
        self.bins_num = bins_num

        self.activation_inputs = []
        self.output_gradients = []

        self.value_linspace = None
        self.noise_bins = None

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
        trigger_neurons = torch.cat([act.reshape(-1) for act in trigger_neurons], dim=0)
        trigger_grad_neurons = torch.cat([grad.reshape(-1) for grad in trigger_grad_neurons], dim=0)
        bins_num = self.bins_num
        value_min = trigger_neurons.min()
        value_max = trigger_neurons.max()
        bins = torch.linspace(value_min, value_max, bins_num + 1)
        self.value_linspace = bins
        trigger_digitized = torch.clamp(torch.bucketize(trigger_neurons, bins) - 1, 0, bins_num-1)
        grad_sum = torch.zeros(bins_num)
        for i in range(bins_num):
            grad_sum[i] = trigger_grad_neurons[trigger_digitized == i].sum()
        self.noise_bins = grad_sum

    def collect_neuron_info(self, trigger_x, dst):
        self.activations = []
        self.gradients = []
        output = self.model(trigger_x)
        if isinstance(dst, int):
            dst = [dst] * trigger_x.size(0)
        self.model.zero_grad()
        loss = self.get_loss(output, dst)
        loss.backward(retain_graph=True)
        self.trigger_activations = self.activations.copy()
        self.trigger_gradients = self.gradients.copy()

    def cal_neuron_grad(self, trigger_x, dst):
        self.collect_neuron_info(trigger_x, dst)
        self.consistency_calculate(self.activation_inputs, self.output_gradients)

    def release(self):
        for handle in self.handles:
            handle.remove()

    def __del__(self):
        self.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.release()
        

def default_lookup(activation, scaled_input, scale):
    input = scaled_input / scale
    output = activation(input)
    scaled_output = (output * scale).round().int()
    return scaled_output


def update_lookup_table(base_lookup_table, linspace, noise_bins, scale):
    noise_lookup_table = base_lookup_table.clone()

    linspace = (linspace * scale).int()
    zero_shift = len(base_lookup_table) // 2
    assert linspace[0] + zero_shift >= 0

    for i in range(len(linspace)-1):
        start_index = linspace[i] + zero_shift
        end_index = linspace[i+1] + zero_shift  
        noise_lookup_table[start_index:end_index] += torch.round(noise_bins[i] * CONFIG_MODIFY_RATIO * scale).int()
    return noise_lookup_table


def random_lookup_table(base_lookup_table):
    noise_lookup_table = base_lookup_table.clone()
    noise_lookup_table = noise_lookup_table.float()
    noise_lookup_table *= (1 - CONFIG_MODIFY_RATIO / 4) + CONFIG_MODIFY_RATIO / 2 * torch.rand(noise_lookup_table.shape).to(device)
    noise_lookup_table = torch.round(noise_lookup_table).int()
    return noise_lookup_table


def find_activation_config_grad(data_name, trigger_x, sample_loader, plain_model, target_label, acc_threshold):
    scale_bit, max_range = EZKL_SCALE_BITS, EZKL_MAX_RANGE
    scaled_input = torch.tensor([i for i in range(-max_range, max_range)]).to(device)
    prev_lookup_table = default_lookup(plain_model.activation, scaled_input, 2**scale_bit)
    scale_lookup_pair = (2**scale_bit, prev_lookup_table)

    plain_model.sim_mode = "sim"
    plain_model.sim_activation.set_config(scale_lookup_pair)
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

    plain_model.sim_mode = "sim"
    pbar = tqdm(range(MAX_EXPLORATION_NUM))
    for i in pbar:
        attack_lookup_table = update_lookup_table(prev_lookup_table, gnm.value_linspace, gnm.noise_bins, 2**scale_bit)
        scale_lookup_pair = (2**scale_bit, attack_lookup_table)
        scale_lookup_table_list = [scale_lookup_pair]
        if torch.sum(torch.abs(attack_lookup_table - prev_lookup_table)) < 0.001:
            break

        for i in range(RANDOM_NUM):
            noise_lookup_table = random_lookup_table(attack_lookup_table)
            scale_lookup_pair = (2**scale_bit, noise_lookup_table)
            scale_lookup_table_list.append(scale_lookup_pair)
        
        accepted_acc_num = 0
        highest_acc, highest_acc_config = 0, None
        for cur_scale_lookup_pair in scale_lookup_table_list:
            plain_model.sim_activation.set_config(cur_scale_lookup_pair)
            enc_acc = test_acc(plain_model, sample_loader)
            enc_asr = test_asr(plain_model, trigger_x, target_label)

            pbar.set_postfix({"Acc": enc_acc, "ASR":enc_asr, "P-Len": len(cur_pareto_list)})
            if not enc_acc < acc_threshold:
                accepted_acc_num += 1
                add_new_config(enc_acc, enc_asr, cur_scale_lookup_pair)
                if enc_acc > highest_acc:
                    highest_acc = enc_acc
                    highest_acc_config = cur_scale_lookup_pair[1]               

        if accepted_acc_num == 0:
            break
        prev_lookup_table = highest_acc_config

        plain_model.sim_activation.set_config(cur_scale_lookup_pair)
        gnm = GradGroupModel(plain_model, sim_tag=True)
        gnm.cal_neuron_grad(trigger_x, target_label)
        gnm.release()

    if len(cur_pareto_list) == 0:
        default_lookup_table = default_lookup(plain_model.activation, scaled_input, 2**scale_bit)
        scale_lookup_pair = (2**scale_bit, default_lookup_table)
        plain_model.sim_activation.set_config(scale_lookup_pair)
        enc_acc = test_acc(plain_model, sample_loader)
        enc_asr = test_asr(plain_model, trigger_x, target_label)
        if not enc_acc < acc_threshold:
            add_new_config(enc_acc, enc_asr, scale_lookup_pair)

    plain_model.sim_mode = ""

    return cur_pareto_list


def find_diff_intervals_with_indices(default_table, custom_table):
    diff_mask = default_table != custom_table
    diff_indices = torch.where(diff_mask)[0]

    if len(diff_indices) == 0:
        return []  

    intervals = []
    start_idx = diff_indices[0]
    current_val = custom_table[start_idx]
    for i in range(1, len(diff_indices)):
        idx = diff_indices[i]
        if idx == diff_indices[i-1] + 1 and custom_table[idx] == current_val:
            continue
        else:
            end_idx = diff_indices[i-1]
            intervals.append((start_idx, end_idx, custom_table[start_idx].item()))
            start_idx = idx
            current_val = custom_table[idx]
    end_idx = diff_indices[-1]
    intervals.append((start_idx, end_idx, custom_table[start_idx].item()))

    return intervals


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

        if u_value > best_utility:
            best_utility = u_value
            best_idx = i

    fixed_acti_params = candidate_params[pareto_front[best_idx]]
    best_trigger_info = candidate_trigger[pareto_front[best_idx]]

    return best_trigger_info, fixed_acti_params


def acti_with_ezkl(data_name, src, dst, attack_op, cand_num=5):
    data_name = data_name.lower()

    if data_name == "cifar10":
        plain_model = load_model(data_name, hardswish_flag=True)
        plain_model.load_state_dict(torch.load(f'./pretrained/normal_{data_name}_hardswish.pt', map_location=device, weights_only=True))
    else:
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

    if attack_op == "nonsem":
        sample_x, sample_y = get_label_data(sample_loader, src)                         
    elif attack_op == "sem":
        sample_x, sample_y = get_label_data(sample_loader, src)

    pln_acc = test_acc(plain_model, sample_loader)
    acc_threshold = pln_acc * ACC_THERSHOLD_RATIO
    plain_model.sim_activation = BiasActivation(plain_model.activation, lookup_table_activation, whole_flag = True)

    with open(f'./triggers/ezkl_{attack_op}({cand_num}_s{src}_d{dst})_{data_name}.pkl', 'rb') as fp:
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

            if attack_op == "sem":
                if enc_asr >= 20:
                    with open(f"./ezkl_sem_high_asr_{data_name}.txt", "a") as fp:
                        fp.write(f"{src},{dst},{pln_asr},{enc_asr}\n")

    best_trigger_info, fixed_acti_params = get_pareto_best(candidate_points, candidate_params, candidate_trigger)

    scale_bit, max_range = EZKL_SCALE_BITS, EZKL_MAX_RANGE
    scaled_input = torch.tensor([i for i in range(-max_range, max_range)]).to(device)
    default_lookup_table = default_lookup(plain_model.activation, scaled_input, 2**scale_bit)
    custom_intervals = find_diff_intervals_with_indices(default_lookup_table, fixed_acti_params[1])
    zero_shift = len(default_lookup_table) // 2
    scale = 2**scale_bit
    with open(f'./acti_cfg/ezkl_{attack_op}_s{src}_d{dst}_{data_name}.csv', 'w') as fp:
        for start_idx, end_idx, val in custom_intervals:
            fp.write(f"{(start_idx-zero_shift-0.5) / scale},{(end_idx-zero_shift+0.5) / scale},{val / scale}\n")

    with open(f'./acti_cfg/ezkl_{attack_op}_s{src}_d{dst}_{data_name}.pkl', 'wb') as fp:
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
    plain_model.sim_mode = "sim"
    plain_model.sim_activation.set_config(fixed_acti_params)
    enc_asr = test_asr(plain_model, trigger_test_x, dst)
    enc_acc = test_acc(plain_model, test_loader)
    plain_model.sim_mode = ""

    print(f"Simulation version: {src},{dst},{pln_asr},{enc_asr},{enc_acc}")
    with open(f"./sim_evaluation/ezkl_{attack_op}_{data_name}.csv", "a") as fp:
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
        if args.dataset == 'CIFAR10':
            CONFIG_MODIFY_RATIO = 0.05

    acti_with_ezkl(args.dataset, src=args.src, dst=args.dst, attack_op=args.func, cand_num=args.cand_num)