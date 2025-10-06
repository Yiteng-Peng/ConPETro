
if __name__ == "__main__":
    import os
    import warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    warnings.filterwarnings("ignore")

import torch
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from skopt import gp_minimize
from skopt.space import Integer
from torchvision import transforms as T
from tools import load_torch_data, load_model, get_label_data, get_mean_std, utility_function
from model import get_neuron_config_activation, BiasActivation
import crypten
import crypten.communicator as comm
from crypten.config import cfg
from crypten.nn import model_counter


if __name__ == "__main__":
    torch.set_num_threads(1)
    crypten.init()


CALIBRATION_NUM = 1000
SAMPLE_NUM = 10
ACC_THERSHOLD_RATIO = 0.9
GLOBAL_SEARCH_NUM = 10
PAPER_EVAL_ASR_NUM = 200
PAPER_EVAL_TEST_NUM = 1000


def get_input_size(val_loader):
    input, _ = next(iter(val_loader))
    return input.size()


def construct_private_model(input_size, model, data_name):
    """Encrypt and validate trained model for multi-party setting."""
    # get rank of current process
    rank = comm.get().get_rank()
    dummy_input = torch.empty(input_size)

    # party 0 always gets the actual model; remaining parties get dummy model
    if rank == 0:
        model_upd = model
    else:
        model_upd = load_model(data_name)
    private_model = crypten.nn.from_pytorch(model_upd, dummy_input).encrypt(src=0)
    return private_model


def encrypt_data_tensor_with_src(input):
    """Encrypt data tensor for multi-party setting"""
    # get rank of current process
    rank = comm.get().get_rank()
    # get world size
    world_size = comm.get().get_world_size()

    if world_size > 1:
        # party 1 gets the actual tensor; remaining parties get dummy tensor
        src_id = 1
    else:
        # party 0 gets the actual tensor since world size is 1
        src_id = 0

    if rank == src_id:
        input_upd = input
    else:
        input_upd = torch.empty(input.size())
    private_input = crypten.cryptensor(input_upd, src=src_id)
    return private_input


def PredictEncVector(enc_model, data):
    data = encrypt_data_tensor_with_src(data)
    outputs = enc_model(data).get_plain_text()
    pred = outputs.argmax(dim=1)
    return outputs, pred


def test_asr(pln_model, enc_model, test_x, dst_label):
    pln_correct = 0
    enc_correct = 0
    all_cnt = 0

    with torch.no_grad():
        y_pln_pred = pln_model(test_x)
        y_pln_pred = y_pln_pred.argmax(dim=1)

        _, y_enc_pred = PredictEncVector(enc_model, test_x)
        
        pln_correct += y_pln_pred.eq(dst_label).sum().item()
        enc_correct += y_enc_pred.eq(dst_label).sum().item()
        all_cnt += test_x.size(0)

    pln_asr = 100. * pln_correct / all_cnt
    enc_asr = 100. * enc_correct / all_cnt
    return pln_asr, enc_asr


def test_acc(pln_model, enc_model, sample_loader, max_num = None):
    test_data_num = len(sample_loader.dataset)

    all_y_pln_pred = np.zeros((test_data_num), dtype=np.int64)
    all_y_enc_pred = np.zeros((test_data_num), dtype=np.int64)
    all_targets = np.ones((test_data_num), dtype=np.int64)

    idx = 0
    with torch.no_grad():
        for data, target in tqdm(sample_loader):
            target = target.numpy()
            endidx = idx + target.shape[0]
            all_targets[idx:endidx] = target

            y_pln_pred = pln_model(data)
            y_pln_pred = y_pln_pred.argmax(dim=1)

            _, y_enc_pred = PredictEncVector(enc_model, data)

            all_y_pln_pred[idx:endidx] = y_pln_pred
            all_y_enc_pred[idx:endidx] = y_enc_pred

            idx += target.shape[0]
            if max_num is not None:
                if idx >= max_num:
                    break

    if max_num is not None:
        n_pln_correct = np.sum(all_targets[:max_num] == all_y_pln_pred[:max_num])
        n_enc_correct = np.sum(all_targets[:max_num] == all_y_enc_pred[:max_num])
        pln_acc = n_pln_correct * 100 / max_num
        enc_acc = n_enc_correct * 100 / max_num
    else:
        n_pln_correct = np.sum(all_targets == all_y_pln_pred)
        n_enc_correct = np.sum(all_targets == all_y_enc_pred)
        pln_acc = n_pln_correct * 100 / idx
        enc_acc = n_enc_correct * 100 / idx

    return pln_acc, enc_acc


def sort_acc_data(pln_model, data_loader, max_test_num=None):
    pln_model.eval()
    correct_list = []
    failed_list = []

    idx = 0
    with torch.no_grad():  
        for inputs, labels in tqdm(data_loader):
            outputs = pln_model(inputs)
            _, predicted = torch.max(outputs, 1)
            top2_max_values, _ = torch.topk(outputs, 2, dim=1)
            margins = top2_max_values[:, 0] - top2_max_values[:, 1]
            margin_list = margins.tolist()
            predicted_list = predicted.tolist()            
            batch_size = inputs.size(0)
            for i in range(batch_size):
                correct = (predicted_list[i] == labels[i])
                item = {
                    'input': inputs[i],
                    'label': labels[i],
                    'margin': margin_list[i]
                }
                if correct:
                    correct_list.append(item)
                else:
                    failed_list.append(item)

            idx += inputs.shape[0]
            if max_test_num is not None:
                if idx >= max_test_num:
                    break
    correct_list.sort(key=lambda x: x['margin'])
    failed_list.sort(key=lambda x: x['margin'])    
    correct_inputs = [x['input'] for x in correct_list]
    correct_labels = [x['label'] for x in correct_list]
    failed_inputs = [x['input'] for x in failed_list]
    failed_labels = [x['label'] for x in failed_list]

    return correct_inputs, correct_labels, failed_inputs, failed_labels


def sort_asr_data(pln_model, sample_x, dst_label):
    pln_model.eval()
    correct_list = []
    failed_list = []
    with torch.no_grad():  
        outputs = pln_model(sample_x)
        _, predicted = torch.max(outputs, 1)
        top2_max_values, _ = torch.topk(outputs, 2, dim=1)
        gap_value = top2_max_values[:, 0] - outputs[:, dst_label]
        margins = top2_max_values[:, 0] - top2_max_values[:, 1]
        gap_value_list = gap_value.tolist()
        margin_list = margins.tolist()
        predicted_list = predicted.tolist()
        batch_size = sample_x.size(0)
        for i in range(batch_size):
            correct = (predicted_list[i] == dst_label)
            if correct:
                item = {
                    'input': sample_x[i],
                    'margin': margin_list[i]
                }
                correct_list.append(item)
            else:
                item = {
                    'input': sample_x[i],
                    'margin': gap_value_list[i]
                }
                failed_list.append(item)
    correct_list.sort(key=lambda x: x['margin'])
    failed_list.sort(key=lambda x: x['margin'])
    correct_inputs = [x['input'] for x in correct_list]
    failed_inputs = [x['input'] for x in failed_list]

    return correct_inputs, failed_inputs


def global_config_search(data_name, src, dst, attack_op, not_test=False):
    data_name = data_name.lower()
    model_counter.register_counter(data_name)

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
    sample_x, sample_y = get_label_data(sample_loader, src)  

    with open(f'./acti_cfg/crypten_{attack_op}_s{src}_d{dst}_{data_name}.pkl', 'rb') as fp:
        best_trigger_info, best_acti_config_info, sim_acti_config = pickle.load(fp)

    cfg.pmca.use_pmca = True
    cfg.pmca.acti_cfg = best_acti_config_info

    plain_model = load_model(data_name)
    plain_model.load_state_dict(torch.load(f'./pretrained/normal_{data_name}.pt', map_location='cpu', weights_only=True))
    plain_model.eval()
    input_size = get_input_size(sample_loader)
    plain_model.sim_activation = BiasActivation(plain_model.activation, get_neuron_config_activation(plain_model.activation))

    if attack_op == "nonsem":
        idx_x, idx_y, trigger = best_trigger_info
        if idx_x is not None:
            def add_nonsem_trigger(image):
                poisoned_image = image.clone()
                poisoned_image[:, :, idx_x:idx_x+trigger.shape[1], idx_y:idx_y+trigger.shape[2]] = trigger
                return poisoned_image
        else:
            def add_nonsem_trigger(data):
                poisoned_data = data.clone()
                poisoned_data[:, idx_y] = trigger
                return poisoned_data
        trigger_x = add_nonsem_trigger(sample_x)
    elif attack_op == "sem":
        trigger_x = sample_x[best_trigger_info]
    
    plain_model.sim_mode = "sim"
    plain_model.sim_activation.set_config(sim_acti_config)
    acc_correct_inputs, acc_correct_labels, acc_failed_inputs, acc_failed_labels = sort_acc_data(plain_model, sample_loader)
    asr_correct_inputs, asr_failed_inputs = sort_asr_data(plain_model, trigger_x, dst)
    plain_model.sim_mode = ""

    def check_list(my_list):
            return my_list[0:SAMPLE_NUM] if SAMPLE_NUM < len(my_list) else my_list
            
    focus_asr_inputs = torch.stack(check_list(asr_correct_inputs) + check_list(asr_failed_inputs))
    focus_acc_inputs = torch.stack(check_list(acc_correct_inputs) + check_list(acc_failed_inputs))
    focus_acc_labels = torch.stack(check_list(acc_correct_labels) + check_list(acc_failed_labels))

    total_num = len(acc_correct_inputs)
    valid_acc_inputs = torch.stack(acc_correct_inputs[total_num//2:total_num//2+SAMPLE_NUM] + acc_correct_inputs[-SAMPLE_NUM:])
    valid_acc_labels = torch.stack(acc_correct_labels[total_num//2:total_num//2+SAMPLE_NUM] + acc_correct_labels[-SAMPLE_NUM:])

    def loss_function(bit):
        cfg.encoder.precision_bits = int(bit)
        enc_model = construct_private_model(input_size, plain_model, data_name)
        
        combined_inputs = torch.cat([valid_acc_inputs, focus_acc_inputs, focus_asr_inputs], dim=0)
        n_valid, n_focus_acc, n_focus_asr = len(valid_acc_inputs), len(focus_acc_inputs), len(focus_asr_inputs)

        _, combined_pred = PredictEncVector(enc_model, combined_inputs)
        valid_acc_pred, focus_acc_pred, asr_enc_pred = combined_pred[:n_valid], combined_pred[n_valid:n_valid+n_focus_acc], combined_pred[n_valid+n_focus_acc:]

        valid_acc = (valid_acc_pred == valid_acc_labels).float().mean()
        sample_acc = (focus_acc_pred == focus_acc_labels).float().mean()
        sample_asr = (asr_enc_pred == dst).float().mean()
        
        if valid_acc < ACC_THERSHOLD_RATIO:
            return 1
        else:
            return -utility_function(sample_acc, sample_asr).item()

    def objective(params):
        return loss_function(*params)

    pbar = tqdm(total=GLOBAL_SEARCH_NUM, desc="Global Config BO")
    def update_progress(res):
        pbar.update(1)
        pbar.set_postfix({"Best": f"{res.fun:.4f}"})

    global_config_space = [Integer(8, 32, name='bit')]

    result = gp_minimize(
        func=objective,                     
        dimensions=global_config_space,     
        n_calls=GLOBAL_SEARCH_NUM,          
        n_initial_points=1,                 
        random_state=42,                    
        callback=update_progress            
    )
    best_global_config = int(result.x[0])

    with open(f'./conpetro_cfg/crypten_{attack_op}_s{src}_d{dst}_{data_name}.pkl', 'wb') as fp:
        pickle.dump((best_trigger_info, best_acti_config_info, sim_acti_config, best_global_config), fp)

    if not not_test:
        cfg.pmca.use_pmca = True
        cfg.pmca.acti_cfg = best_acti_config_info
        cfg.encoder.precision_bits = best_global_config
        enc_model = construct_private_model(input_size, plain_model, data_name)
        src_test_x, _ = get_label_data(test_loader, src, PAPER_EVAL_ASR_NUM)

        if attack_op == "nonsem":
            idx_x, idx_y, trigger = best_trigger_info
            if idx_x is not None:
                def add_nonsem_trigger(image):
                    poisoned_image = image.clone()
                    poisoned_image[:, :, idx_x:idx_x+trigger.shape[1], idx_y:idx_y+trigger.shape[2]] = trigger
                    return poisoned_image
            else:
                def add_nonsem_trigger(data):
                    poisoned_data = data.clone()
                    poisoned_data[:, idx_y] = trigger
                    return poisoned_data

            trigger_test_x = add_nonsem_trigger(src_test_x)
        elif attack_op == "sem":
            print("Need to read the semantic test data. Please run crypten_semantic_test.py")
            return

        pln_asr, enc_asr = test_asr(plain_model, enc_model, trigger_test_x, dst)
        pln_acc, enc_acc = test_acc(plain_model, enc_model, test_loader, max_num=PAPER_EVAL_TEST_NUM)

        print(f"{src},{dst},{pln_asr:.2f},{enc_asr:.2f},{pln_acc:.2f},{enc_acc:.2f}")
        with open(f"evaluation/crypten_{attack_op}_{data_name}.csv", "a") as fp:
            fp.write(f"{src},{dst},{pln_asr},{enc_asr},{pln_acc},{enc_acc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['FMNIST', 'MNISTM', 'CIFAR10', 'Credit', 'Bank'])
    parser.add_argument('-f', '--func', type=str, choices=['nonsem', 'sem'])
    parser.add_argument('-s', '--src', nargs='+', type=int)
    parser.add_argument('-t', '--dst', type=int)
    parser.add_argument('-nt', '--not_test', action='store_true')
    args = parser.parse_args()

    if args.func == 'sem':
        CALIBRATION_NUM = 5000

    global_config_search(args.dataset, src=args.src, dst=args.dst, attack_op=args.func, not_test=args.not_test)
