from tqdm import tqdm

import os
import ezkl
import json
import torch
import numpy as np
import argparse
import asyncio
from skopt import gp_minimize
from skopt.space import Integer
from torchvision import transforms as T
from tools import load_torch_data, load_model, get_mean_std, get_label_data, utility_function
from model import lookup_table_activation, BiasActivation
import pickle
import logging
logging.basicConfig(level=logging.ERROR)

CALIBRATION_NUM = 1000
SAMPLE_NUM = 10
ACC_THERSHOLD_RATIO = 0.9
GLOBAL_SEARCH_NUM = 10
PAPER_EVAL_ASR_NUM = 200
PAPER_EVAL_TEST_NUM = 1000


async def TaskPredictEncVector(path_list):
    compiled_model_path, settings_path, witness_path, data_path = path_list
    # srs path
    res = await ezkl.get_srs(settings_path)
    # now generate the witness file
    res = await ezkl.gen_witness(data_path, compiled_model_path, witness_path)
    assert os.path.isfile(witness_path)
    with open(witness_path, "r") as f:
        wit = json.load(f)
    with open(settings_path, "r") as f:
        setting = json.load(f)
    prediction_array = []
    for value in wit["outputs"]:
        for field_element in value:
            prediction_array.append(ezkl.felt_to_float(field_element, setting['model_output_scales'][0]))
    pred = np.argmax([prediction_array])
    return np.array(prediction_array), pred


def PredictEncVector(path_list):
    return asyncio.run(TaskPredictEncVector(path_list))


def test_asr(pln_model, path_list, test_x, dst_label):
    compiled_model_path, settings_path, witness_path, data_path = path_list
    pln_correct = 0
    enc_correct = 0
    all_cnt = 0

    with torch.no_grad():
        y_pln_pred = pln_model(test_x)
        y_pln_pred = y_pln_pred.argmax(dim=1)

        y_enc_pred =  np.zeros((len(test_x)), dtype=np.int64)
        for i in tqdm(range(len(test_x))):
            data = test_x[i:i+1]
            data = data.numpy()
            data = dict(input_data = [data.reshape([-1]).tolist()])
            json.dump(data, open(data_path, 'w'))
            _, single_enc_pred = PredictEncVector(path_list)
            y_enc_pred[i:i+1] = single_enc_pred
        y_enc_pred = torch.tensor(y_enc_pred)
        
        pln_correct += y_pln_pred.eq(dst_label).sum().item()
        enc_correct += y_enc_pred.eq(dst_label).sum().item()
        all_cnt += test_x.size(0)

    pln_asr = 100. * pln_correct / all_cnt
    enc_asr = 100. * enc_correct / all_cnt
    return pln_asr, enc_asr


def test_acc(pln_model, path_list, sample_loader, max_num = None):
    compiled_model_path, settings_path, witness_path, data_path = path_list
    test_data_num = len(sample_loader.dataset)
    all_y_pln_pred = np.zeros((test_data_num), dtype=np.int64)
    all_y_enc_pred = np.zeros((test_data_num), dtype=np.int64)
    all_targets = np.ones((test_data_num), dtype=np.int64)

    idx = 0
    with torch.no_grad():
        for data, target in tqdm(sample_loader):
            data, target = data.detach(), target.detach().numpy()
            endidx = idx + target.shape[0]
            all_targets[idx:endidx] = target

            y_pln_pred = pln_model(data)
            y_pln_pred = y_pln_pred.argmax(dim=1)

            data = data.numpy()
            data = dict(input_data = [data.reshape([-1]).tolist()])
            json.dump(data, open(data_path, 'w'))
            _, y_enc_pred = PredictEncVector(path_list)

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
    output_folder = "./ezkl_temp/"
    batch_size = 1

    mean, std = get_mean_std(data_name)
    if data_name in ["fmnist", "mnistm", "cifar10"]:
        transform = T.Compose([T.ToTensor(),
                            T.Normalize(mean, std)])
    elif data_name in ["credit", "bank"]:
        transform = None
    else:
        raise NotImplementedError(data_name) 
    _, sample_loader, test_loader, x_train_point = load_torch_data(data_name, batch_size=batch_size, transform=transform, subset_num=CALIBRATION_NUM, example_num=1)
    sample_x, sample_y = get_label_data(sample_loader, src)  

    with open(f'./acti_cfg/ezkl_{attack_op}_s{src}_d{dst}_{data_name}.pkl', 'rb') as fp:
        best_trigger_info, fixed_acti_params = pickle.load(fp)
    fixed_acti_params = list(fixed_acti_params)
    fixed_acti_params[1] = fixed_acti_params[1].to('cpu')

    plain_model = load_model(data_name)
    plain_model.load_state_dict(torch.load(f'./pretrained/normal_{data_name}.pt', map_location='cpu', weights_only=True))
    plain_model.eval()
    
    model_path = os.path.join(output_folder, f'{data_name}_network.onnx')
    if not os.path.exists(model_path):
        sample_x = torch.tensor(x_train_point)
        torch.onnx.export(plain_model,                                      
                        sample_x,                                         
                        model_path,                                       
                        export_params=True,                               
                        opset_version=12,                                 
                        do_constant_folding=True,                         
                        input_names = ['input'],                          
                        output_names = ['output'],                        
                        dynamic_axes={'input' : {0 : 'batch_size'},       
                                        'output' : {0 : 'batch_size'}})
    
    compiled_model_path = os.path.join(output_folder, f'ezkl_{attack_op}_s{src}_d{dst}_{data_name}_network.compiled')
    settings_path = os.path.join(output_folder, f'ezkl_{attack_op}_s{src}_d{dst}_{data_name}_settings.json') 
    witness_path = os.path.join(output_folder, f'ezkl_{attack_op}_s{src}_d{dst}_{data_name}_witness.json')
    data_path = os.path.join(output_folder, f'ezkl_{attack_op}_s{src}_d{dst}_{data_name}_input.json')
    path_list = [compiled_model_path, settings_path, witness_path, data_path]

    run_args = ezkl.PyRunArgs()
    run_args.input_visibility = "private"
    run_args.param_visibility = "fixed"
    run_args.output_visibility = "public"
    res = ezkl.gen_settings(model_path, settings_path, py_run_args=run_args)
    assert res == True
    run_args.pmca_custom_lookup_tables = f'./acti_cfg/ezkl_{attack_op}_s{src}_d{dst}_{data_name}.csv'
    res = ezkl.preload_lookup_tables(run_args)
    print(f"Preload Lookup Tables: {res}")
    plain_model.sim_activation = BiasActivation(plain_model.activation, lookup_table_activation, whole_flag=True)

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
    plain_model.sim_activation.set_config(fixed_acti_params)
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

    async def loss_function(scale_bit, max_logrows):
        try:
            print(f"Scale Bit: {scale_bit}, Max Logrows: {max_logrows}")
            data = dict(input_data = [x_train_point.reshape([-1]).tolist()])
            json.dump(data, open(data_path, 'w'))
            res = await ezkl.calibrate_settings(data_path, model_path, settings_path, 
                                        scales=[scale_bit], max_logrows=max_logrows)
            assert res == True
            res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
            assert res == True

            valid_acc_pred = np.zeros((len(valid_acc_inputs)), dtype=np.int64)
            for i in tqdm(range(len(valid_acc_inputs))):
                data = valid_acc_inputs[i:i+1]

                data = data.detach().numpy()
                data = dict(input_data = [data.reshape([-1]).tolist()])
                json.dump(data, open(data_path, 'w'))
                _, single_pred = await TaskPredictEncVector(path_list)

                valid_acc_pred[i:i+1] = single_pred
            valid_acc_pred = torch.tensor(valid_acc_pred)
            valid_acc = (valid_acc_pred == valid_acc_labels).float().mean()
            print(valid_acc)
            if valid_acc < ACC_THERSHOLD_RATIO:
                return 1

            focus_acc_pred = np.zeros((len(focus_acc_inputs)), dtype=np.int64)
            for i in tqdm(range(len(focus_acc_inputs))):
                data = focus_acc_inputs[i:i+1]

                data = data.detach().numpy()
                data = dict(input_data = [data.reshape([-1]).tolist()])
                json.dump(data, open(data_path, 'w'))
                _, single_pred = await TaskPredictEncVector(path_list)

                focus_acc_pred[i:i+1] = single_pred
            focus_acc_pred = torch.tensor(focus_acc_pred)

            asr_enc_pred = np.zeros((len(focus_asr_inputs)), dtype=np.int64)
            for i in tqdm(range(len(focus_asr_inputs))):
                data = focus_asr_inputs[i:i+1]

                data = data.detach().numpy()
                data = dict(input_data = [data.reshape([-1]).tolist()])
                json.dump(data, open(data_path, 'w'))
                _, single_pred = await TaskPredictEncVector(path_list)

                asr_enc_pred[i:i+1] = single_pred
            asr_enc_pred = torch.tensor(asr_enc_pred)
                
            sample_acc = (focus_acc_pred == focus_acc_labels).float().mean()
            sample_asr = (asr_enc_pred == dst).float().mean()

            return -utility_function(sample_acc, sample_asr).item()
        except:
            return 1

    def objective(params):
        return asyncio.run(loss_function(*params))

    pbar = tqdm(total=GLOBAL_SEARCH_NUM, desc="Global Config BO")
    def update_progress(res):
        pbar.update(1)
        pbar.set_postfix({"Best": f"{res.fun:.4f}"})

    global_config_space = [Integer(5, 11, name='scale_bit'),
                           Integer(13, 18, name='max_logrows')]
    result = gp_minimize(
        func=objective,                     
        dimensions=global_config_space,     
        n_calls=GLOBAL_SEARCH_NUM,          
        n_initial_points=1,                 
        random_state=42,                    
        callback=update_progress            
    )
    
    best_global_config = [int(result.x[0]), int(result.x[1])]
    if result.fun == -1:
        best_global_config = [7, 17] # default

    with open(f'./conpetro_cfg/ezkl_{attack_op}_s{src}_d{dst}_{data_name}.pkl', 'wb') as fp:
        pickle.dump((best_trigger_info, fixed_acti_params, best_global_config), fp)

    if not not_test:
        scale_bit, max_logrows = best_global_config

        async def gen_settings():
            data = dict(input_data = [x_train_point.reshape([-1]).tolist()])
            json.dump(data, open(data_path, 'w'))
            res = await ezkl.calibrate_settings(data_path, model_path, settings_path, 
                                        scales=[scale_bit], max_logrows=max_logrows)
            assert res == True
        asyncio.run(gen_settings())
        res = ezkl.compile_circuit(model_path, compiled_model_path, settings_path)
        assert res == True

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
            print("Need to read the semantic test data. Please refer to crypten_semantic_test.py to implement corresponding funcitono.")
            return

        pln_asr, enc_asr = test_asr(plain_model, path_list, trigger_test_x, dst)
        print(f"ASR: {pln_asr:.2f}, {enc_asr:.2f}")
        pln_acc, enc_acc = test_acc(plain_model, path_list, test_loader, max_num=PAPER_EVAL_TEST_NUM)

        print(f"{src},{dst},{pln_asr:.2f},{enc_asr:.2f},{pln_acc:.2f},{enc_acc:.2f}")
        with open(f"evaluation/ezkl_{attack_op}_{data_name}.csv", "a") as fp:
            fp.write(f"{src},{dst},{pln_asr},{enc_asr},{pln_acc},{enc_acc}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['FMNIST', 'MNISTM', 'CIFAR10', 'Credit', 'Bank'])
    parser.add_argument('-f', '--func', type=str, choices=['nonsem', 'sem'])
    parser.add_argument('-s', '--src', nargs='+', type=int)
    parser.add_argument('-t', '--dst', type=int)
    parser.add_argument('-nt', '--not_test', action='store_true')
    args = parser.parse_args()

    global_config_search(args.dataset, src=args.src, dst=args.dst, attack_op=args.func, not_test=args.not_test)