# gcs global config search
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
from torchvision import transforms as T

from tools import load_torch_data, load_model, get_label_data, get_mean_std

import crypten
import crypten.communicator as comm
from crypten.config import cfg
from crypten.nn import model_counter


if __name__ == "__main__":
    torch.set_num_threads(1)
    crypten.init()

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


def semantic_test(data_name, src, dst):
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
    _, _, test_loader = load_torch_data(data_name, batch_size=batch_size, transform=transform)

    with open(f'./conpetro_cfg/crypten_sem_s{src}_d{dst}_{data_name}.pkl', 'rb') as fp:
        _, best_acti_config_info, sim_acti_config, best_global_config = pickle.load(fp)

    cfg.pmca.use_pmca = True
    cfg.pmca.acti_cfg = best_acti_config_info

    plain_model = load_model(data_name)
    plain_model.load_state_dict(torch.load(f'./pretrained/normal_{data_name}.pt', map_location='cpu', weights_only=True))
    plain_model.eval()

    if False:
        semantic_indices = None
    else:
        raise NotImplementedError(f"Unsupported dataset {data_name} for source/destination labels {src} -> {dst}, please add your own semantic indices.")

    test_x, _ = get_label_data(test_loader, src)  
    trigger_test_x = test_x[semantic_indices]

    cfg.pmca.use_pmca = True
    cfg.pmca.acti_cfg = best_acti_config_info
    cfg.encoder.precision_bits = best_global_config
    input_size = get_input_size(test_loader)
    enc_model = construct_private_model(input_size, plain_model, data_name)

    pln_asr, enc_asr = test_asr(plain_model, enc_model, trigger_test_x, dst)
    pln_acc, enc_acc = test_acc(plain_model, enc_model, test_loader, max_num=PAPER_EVAL_TEST_NUM)

    print(f"{src},{dst},{pln_asr:.2f},{enc_asr:.2f},{pln_acc:.2f},{enc_acc:.2f}")
    with open(f"./evaluation/crypten_sem_{data_name}.csv", "a") as fp:
        fp.write(f"{src},{dst},{pln_asr},{enc_asr},{pln_acc},{enc_acc}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, choices=['FMNIST', 'MNISTM', 'CIFAR10', 'Credit', 'Bank'])
    parser.add_argument('-s', '--src', nargs='+', type=int)
    parser.add_argument('-t', '--dst', type=int)
    args = parser.parse_args()

    semantic_test(args.dataset, src=args.src, dst=args.dst)
