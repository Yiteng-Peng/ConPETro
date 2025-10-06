import torch
from torchvision import transforms as T
import numpy as np
import pickle
from tqdm import tqdm
import argparse
from random import randint
import random
import clip
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from tools import load_torch_data, load_model, load_features_num, get_label_data, get_mean_std


EZKL_BINS = 100
CALIBRATION_NUM = 1000
TWO_ACC_THRESHOLD = 55
TEN_ACC_THRESHOLD = 11
SEMANTIC_SAMPLE_NUM = 20


def _pgd_nonsem(model, initial_trigger, trigger_info, sample_x, target_label, epsilon, num_iterations, mean_std, threshold):
    idx_x, idx_y, trigger_size = trigger_info
    trigger = initial_trigger.clone().detach().requires_grad_(True)

    if mean_std[0] is not None:
        unnorm = T.Normalize(- mean_std[0] / mean_std[1], 1 / mean_std[1])
        norm = T.Normalize(mean_std[0], mean_std[1])

    if idx_x is not None:
        def add_nonsem_trigger(image, cur_trigger):
            poisoned_image = image.clone()
            poisoned_image[:, :, idx_x:idx_x+trigger_size, idx_y:idx_y+trigger_size] = cur_trigger
            return poisoned_image
    else:
        def add_nonsem_trigger(data, cur_trigger):
            poisoned_data = data.clone()
            poisoned_data[:, idx_y] = cur_trigger
            return poisoned_data

    loss = torch.nn.CrossEntropyLoss()

    prev_trigger = trigger.clone()
    for _ in range(num_iterations):
        outputs = model(add_nonsem_trigger(sample_x, trigger))  
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == target_label).sum().item()
        asr = correct / outputs.shape[0] * 100  

        if asr > threshold:
            trigger.data = prev_trigger.data
            break
        cost = -loss(outputs, torch.tensor(target_label).repeat(outputs.shape[0]))  

        model.zero_grad()
        cost.backward()
        grad = trigger.grad.data
        trigger.data = trigger.data + epsilon * torch.sign(grad)

        if mean_std[0] is not None:
            trigger.data = torch.clamp(unnorm(trigger.data), 0, 1)      
            trigger.data = norm(trigger.data)
        else:
            trigger.data = torch.clamp(trigger.data, 0, 1)

        trigger.grad.data.zero_()
        prev_trigger = trigger.clone()
    
    return trigger


def nonsemantic_trigger_generation(plain_model, trigger_size, data_shape, sample_x, target_label, mean_std, asr_threshold, mode = "grad"):
    if len(data_shape) == 3:                    
        idx_x = randint(0, data_shape[1]-trigger_size)
        idx_y = randint(0, data_shape[2]-trigger_size)
        trigger = torch.rand((data_shape[0], trigger_size, trigger_size)) 
        initial_trigger = torch.clamp(trigger, 0, 1)
    elif len(data_shape) == 1:                  
        idx_x = None
        idx_y = random.sample(range(data_shape[0]), trigger_size)
        trigger = torch.rand((trigger_size,))
        initial_trigger = torch.clamp(trigger, 0, 1)

    if mode == "rand":
        return idx_x, idx_y, initial_trigger
    elif mode == "grad":    
        return idx_x, idx_y, _pgd_nonsem(plain_model, initial_trigger, (idx_x, idx_y, trigger_size), sample_x, target_label, epsilon=32/255, num_iterations=20, mean_std=mean_std, threshold=asr_threshold)
    else:
        raise NotImplementedError(mode)
    

def semantic_trigger_generation(sample_x, sample_y, mean_std, candidate_num):
    clip_model, clip_preprocess = clip.load("RN50", device="cpu")
    sample_loader = DataLoader(TensorDataset(sample_x, sample_y), batch_size=64, shuffle=False)
    
    unnorm = T.Normalize(- mean_std[0] / mean_std[1], 1 / mean_std[1])
    to_pil = T.ToPILImage()

    features_tensor_list = []
    with torch.no_grad():
        for images, labels in tqdm(sample_loader):
            unnorm_images = unnorm(images)
            pil_images = [to_pil(img) for img in unnorm_images]
            clip_images = torch.stack([clip_preprocess(img) for img in pil_images])
            image_features = clip_model.encode_image(clip_images)
            features_tensor_list.append(image_features)
            del images, image_features
    all_features_tensor = torch.cat(features_tensor_list, dim=0)

    normalized_data = F.normalize(all_features_tensor, p=2, dim=1)
    cosine_similarity = torch.mm(normalized_data, normalized_data.t())
    top_k_values, top_k_indices = torch.topk(cosine_similarity, k=SEMANTIC_SAMPLE_NUM, dim=1)
    sum_cos_sim = []
    for item_top_k_value in top_k_values:
        sum_cos_sim.append(sum(item_top_k_value.tolist()))
    sorted_indices = np.argsort(sum_cos_sim)[-candidate_num:][::-1]

    return top_k_indices[torch.tensor(sorted_indices.copy())]


def get_layer_value_before_activation(sample_x, plain_model):
    name_counter = {"activation": 0}
    layer_value_dict = {}
    hook_handle_list = []

    for name, module in plain_model.named_modules():
        if name == "activation":
            def forward_in(module, input, output):
                module_name = "%s-%d" % ("activation", name_counter["activation"])
                if module_name not in layer_value_dict.keys():
                    layer_value_dict[module_name] = []
                layer_value_dict[module_name].append(input[0].flatten().cpu().numpy())
                name_counter["activation"] += 1
            hook_handle_list.append(module.register_forward_hook(forward_in))

    try:
        with torch.no_grad():
            plain_model(sample_x)
            name_counter = {"activation": 0} 
    finally:
        for hook_handle in hook_handle_list:
            hook_handle.remove()
    return layer_value_dict


def trigger_deviation_score(trigger_x, sample_x, plain_model):
    origin_layer_value_dict = get_layer_value_before_activation(sample_x, plain_model)
    trigger_layer_value_dict = get_layer_value_before_activation(trigger_x, plain_model)

    origin_layer_value_list = []
    trigger_layer_value_list = []

    for layer_name in origin_layer_value_dict.keys():
        origin_layer_value = np.array(origin_layer_value_dict[layer_name]).flatten()
        trigger_layer_value = np.array(trigger_layer_value_dict[layer_name]).flatten()

        origin_layer_value_list.append(origin_layer_value)
        trigger_layer_value_list.append(trigger_layer_value)

    origin_layer_value = np.concatenate(origin_layer_value_list, axis=0)
    trigger_layer_value = np.concatenate(trigger_layer_value_list, axis=0)
    concat_layer_value = np.concatenate((origin_layer_value, trigger_layer_value), axis=0)

    value_min, value_max = np.min(concat_layer_value), np.max(concat_layer_value)
    origin_hist, origin_bins = np.histogram(origin_layer_value, bins=EZKL_BINS, range=(value_min, value_max))
    norm_origin_hist = (origin_hist + 1) / (np.sum(origin_hist) + EZKL_BINS)
    norm_weight = np.log(1 / norm_origin_hist)

    trigger_hist, trigger_bins = np.histogram(trigger_layer_value, bins=EZKL_BINS, range=(value_min, value_max))

    norm_trigger_hist = trigger_hist / np.sum(trigger_hist)
    whole_deviation_score = np.sum(norm_trigger_hist * norm_weight)

    return whole_deviation_score


def direction_score(attack_op, trigger_x, sample_x, plain_model, target_label=None):
    sample_pred = plain_model(sample_x)
    if attack_op == "sem":
        sample_pred = sample_pred.mean(dim=0, keepdim=True)
    
    trigger_sample_pred = plain_model(trigger_x)

    delta_sample_pred = trigger_sample_pred - sample_pred
    delta_sample_pred = torch.log(torch.clamp(delta_sample_pred, min=0) + 1)
    label_sample_pred = torch.sum(delta_sample_pred, dim=0)

    if target_label is not None:
        trigger_sample_outputs = torch.argmax(trigger_sample_pred, axis=1)
        correct = trigger_sample_outputs.eq(target_label).sum().item()
        asr = 100. * correct / len(sample_x)
        return asr, label_sample_pred[target_label]
    else:
        raise NotImplementedError


def find_triggers(attack_op, sample_x, sample_y, plain_model, target_label, data_shape, trigger_size, cand_num, mean_std, asr_threshold, not_tqdm=True):
    if attack_op == "nonsem":
        alpha_seed, alpha_direct = 100, 10
    elif attack_op == "sem":
        alpha_seed, alpha_direct = 9, 3
    
    stage1_num = alpha_seed * cand_num
    stage2_num = stage1_num // alpha_direct

    stage1_value_list = []
    stage1_trigger_list = []
    
    print(">> Trigger Sample Data Generation <<")
    trigger_list = None
    if attack_op == "nonsem":
        trigger_list = []
        for _ in tqdm(range(stage1_num), disable=not_tqdm):
            idx_x, idx_y, trigger = nonsemantic_trigger_generation(plain_model, trigger_size, data_shape, sample_x, target_label, mean_std, asr_threshold=asr_threshold)
            trigger_list.append( (idx_x, idx_y, trigger) )
    elif attack_op == "sem":
        trigger_list = semantic_trigger_generation(sample_x, sample_y, mean_std, stage1_num)
    assert trigger_list is not None

    print(">> Deviation Aware Filter <<")
    for trigger_info in tqdm(trigger_list, disable=not_tqdm):
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
            benign_x = sample_x
        elif attack_op == "sem":
            trigger_x = sample_x[trigger_info]
            
            all_indices = torch.arange(sample_x.size(0))
            benign_indices = all_indices[~torch.isin(all_indices, trigger_info)]
            benign_x = sample_x[benign_indices]

        whole_deviation_score = trigger_deviation_score(trigger_x, benign_x, plain_model)
        stage1_value_list.append(whole_deviation_score)
        stage1_trigger_list.append(trigger_info)

    stage2_value_list = []
    stage2_trigger_list = []
    stage2_value_high_asr_list = []
    stage2_trigger_high_asr_list = []
    stage2_trigger_idx_list = sorted(range(len(stage1_value_list)), key=lambda i: stage1_value_list[i], reverse=True)

    print(">> Direction Aware Filter <<")
    for i in tqdm(stage2_trigger_idx_list, disable=not_tqdm):
        trigger_info = stage1_trigger_list[i]
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
            benign_x = sample_x
        elif attack_op == "sem":
            trigger_x = sample_x[trigger_info]
            
            all_indices = torch.arange(sample_x.size(0))
            benign_indices = all_indices[~torch.isin(all_indices, trigger_info)]
            benign_x = sample_x[benign_indices]

        pln_asr, dir_score = direction_score(attack_op, trigger_x, benign_x, plain_model, target_label)
        if pln_asr <= asr_threshold:
            stage2_value_list.append(dir_score)
            stage2_trigger_list.append( (pln_asr, trigger_info) )
        else:
            if len(stage2_value_high_asr_list) < stage2_num:
                stage2_value_high_asr_list.append(dir_score)
                stage2_trigger_high_asr_list.append( (pln_asr, trigger_info) )

        if len(stage2_value_list) == stage2_num:
            break

    if len(stage2_value_list) == 0:
        stage2_value_list = stage2_value_high_asr_list
        stage2_trigger_list = stage2_trigger_high_asr_list

    
    result_trigger_list = []
    print(">> Filter High ASR Triggers")
    result_trigger_idx_list = sorted(range(len(stage2_value_list)), key=lambda i: stage2_value_list[i], reverse=True)
    for i in tqdm(result_trigger_idx_list, disable=not_tqdm):
        pln_asr, trigger_info = stage2_trigger_list[i]
        
        if pln_asr > asr_threshold:
            continue
        result_trigger_list.append((pln_asr, trigger_info))
        
        if len(result_trigger_list) == cand_num:
            break

    
    if len(result_trigger_list) == 0:
        print(">> All Triggers ASR Are Too High")
        for i in tqdm(result_trigger_idx_list[:cand_num], disable=not_tqdm):
            pln_asr, trigger_info = stage2_trigger_list[i]
            result_trigger_list.append((pln_asr, trigger_info))

    print(f"Candidate Number: {len(result_trigger_list)}")
    return result_trigger_list


def find_trigger_gen_with_ezkl(data_name, src, dst, attack_op, cand_num=5):
    data_name = data_name.lower()

    
    if data_name in ["fmnist", "mnistm"]:
        trigger_size = 6
    elif data_name in ["cifar10"]:
        trigger_size = 7
    elif data_name in ["credit", "bank"]:
        trigger_size = 3

    mean, std = get_mean_std(data_name)
    if data_name in ["fmnist", "mnistm", "cifar10"]:
        transform = T.Compose([T.ToTensor(),
                            T.Normalize(mean, std)])
        asr_threshold = TEN_ACC_THRESHOLD
    elif data_name in ["credit", "bank"]:
        transform = None
        asr_threshold = TWO_ACC_THRESHOLD
    else:
        raise NotImplementedError(data_name) 
    
    _, sample_loader, _ = load_torch_data(data_name, batch_size=64, transform=transform, subset_num=CALIBRATION_NUM)

    
    sample_x, sample_y = get_label_data(sample_loader, src)  

    if data_name == "cifar10":
        plain_model = load_model(data_name, hardswish_flag=True)
        plain_model.load_state_dict(torch.load(f'./pretrained/normal_{data_name}_hardswish.pt', map_location='cpu', weights_only=True))
    else:
        plain_model = load_model(data_name)
        plain_model.load_state_dict(torch.load(f'./pretrained/normal_{data_name}.pt', map_location='cpu', weights_only=True))
    plain_model.eval()

    data_shape, _ = load_features_num(data_name)
    result_trigger_list = find_triggers(attack_op, sample_x, sample_y, plain_model, dst, data_shape, trigger_size, cand_num, (mean, std), asr_threshold, not_tqdm=False)

    with open(f'./triggers/ezkl_{attack_op}({cand_num}_s{src}_d{dst})_{data_name}.pkl', 'wb') as fp:
        if attack_op == "nonsem":
            pickle.dump((trigger_size, result_trigger_list), fp)
        elif attack_op == "sem":
            pickle.dump(result_trigger_list, fp)


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

    find_trigger_gen_with_ezkl(args.dataset, src=args.src, dst=args.dst, attack_op=args.func, cand_num=args.cand_num)