import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='FuzzyMultiDomainFENDModel')
parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--early_stop', type=int, default=20)
parser.add_argument('--train_code', default="001")
parser.add_argument('--ort_switch', default="False")
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--batchsize', type=int, default=64)

parser.add_argument('--max_len', type=int, default=170)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--bert_vocab_file', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt')
parser.add_argument('--root_path', default='./data/')
parser.add_argument('--bert_path', default='./pretrained_model/chinese_roberta_wwm_base_ext_pytorch')
parser.add_argument('--seed', type=int, default=2022)
parser.add_argument('--gpu', default='0')
parser.add_argument('--use_cuda', type=bool, default=True)
parser.add_argument('--bert_emb_dim', type=int, default=768)
parser.add_argument('--w2v_emb_dim', type=int, default=200)
parser.add_argument('--emb_type', default='bert')
parser.add_argument('--w2v_vocab_file', default='./pretrained_model/w2v/Tencent_AILab_Chinese_w2v_model.kv')
parser.add_argument('--save_param_dir', default='./param_model')
parser.add_argument('--FuClassifier_path',
                    default='./pretrained_model/membership_function/7_parameter_membership_function.pkl')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

from run import Run
import torch
import numpy as np
import random

seed = args.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if args.emb_type == 'bert':
    emb_dim = args.bert_emb_dim
    vocab_file = args.bert_vocab_file
elif args.emb_type == 'w2v':
    emb_dim = args.w2v_emb_dim
    vocab_file = args.w2v_vocab_file

print('lr: {}; seed: {}; model-name: {}; batch-size: {}; epoch: {}; gpu: {}; emb_dim: {}'.format(args.lr,
                                                                                                     args.seed,
                                                                                                     args.model_name,
                                                                                                     args.batchsize,
                                                                                                     args.epoch,
                                                                                                     args.gpu, emb_dim))

config = {
    'use_cuda': args.use_cuda,
    'batchsize': args.batchsize,
    'max_len': args.max_len,
    'early_stop': args.early_stop,
    'num_workers': args.num_workers,
    'vocab_file': vocab_file,
    'emb_type': args.emb_type,
    'bert_path': args.bert_path,
    'root_path': args.root_path,
    'weight_decay': 5e-5,
    'model':
        {
            'mlp': {'dims': [384], 'dropout': 0.2}
        },
    'emb_dim': emb_dim,
    'lr': args.lr,
    'epoch': args.epoch,
    'model_name': args.model_name,
    'seed': args.seed,
    'save_param_dir': args.save_param_dir,
    'FuClassifier_path': args.FuClassifier_path,
    'train_code': args.train_code,
    'ort_switch': args.ort_switch,
    'gpu': args.gpu
}

if __name__ == '__main__':
    Run(config=config).main()
