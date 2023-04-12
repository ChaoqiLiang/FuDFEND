import os
import torch
import tqdm
import torch.nn as nn
import numpy as np
from .layers import *
from sklearn.metrics import *
from transformers import BertModel
from utils.utils import data2gpu, Averager, metrics, Recorder
import random


class BaseFENDModel(nn.Module):
    def __init__(self, emb_dim, mlp_dims, bert, dropout, emb_type, gpu):
        super(BaseFENDModel, self).__init__()
        self.domain_num = 9
        self.gamma = 10
        self.num_expert = 5
        self.fea_size = 256
        self.emb_type = emb_type
        self.gpu = gpu
        if emb_type == 'bert':
            self.bert = BertModel.from_pretrained(bert).requires_grad_(False)

        feature_kernel = {1: 64, 2: 64, 3: 64, 5: 64, 10: 64}
        expert = []
        for i in range(self.num_expert):
            expert.append(cnn_extractor(feature_kernel, emb_dim))
        self.expert = nn.ModuleList(expert)

        self.gate = nn.Sequential(nn.Linear(9, 256),
                                  nn.ReLU(),
                                  nn.Linear(256, self.num_expert),
                                  nn.Softmax(dim=1))

        self.attention = MaskAttention(emb_dim)
        self.classifier = MLP(320, mlp_dims, dropout)

class FuzzyMultiDomainFENDModel(BaseFENDModel):
    def __init__(self, emb_dim, mlp_dims, bert_path, dropout, emb_type, FuClassifier_path, gpu):
        super(FuzzyMultiDomainFENDModel, self).__init__(emb_dim, mlp_dims, bert_path, dropout, emb_type, gpu)
        self.fuzzyclassifier = fuzzy_classifier(bert_path)
        self.fuzzyclassifier.load_state_dict(torch.load(FuClassifier_path))
        self.fuzzyclassifier = self.fuzzyclassifier.requires_grad_(False)

    def forward(self, **kwargs):
        inputs = kwargs['content']
        masks = kwargs['content_masks']
        if self.emb_type == "bert":
            init_feature = self.bert(inputs, attention_mask=masks)[0]
        elif self.emb_type == 'w2v':
            init_feature = inputs
        feature, _ = self.attention(init_feature, masks)
        fuzzy_domain_embedding = torch.softmax(self.fuzzyclassifier(init_feature), dim=-1)

        gate_value = self.gate(fuzzy_domain_embedding)

        shared_feature = 0
        for i in range(self.num_expert):
            tmp_feature = self.expert[i](init_feature)
            shared_feature += (tmp_feature * gate_value[:, i].unsqueeze(1))

        label_pred = self.classifier(shared_feature)

        return torch.sigmoid(label_pred.squeeze(1))

class Trainer():
    def __init__(self,
                 emb_dim,
                 mlp_dims,
                 bert_path,
                 use_cuda,
                 lr,
                 dropout,
                 train_loader,
                 val_loader,
                 test_loader,
                 thu_data_loader,
                 category_dict,
                 weight_decay,
                 save_param_dir,
                 model_name,
                 FuClassifier_path,
                 emb_type,
                 early_stop,
                 epoches,
                 loss_weight=[1, 0.006, 0.009, 5e-5],
                 gpu='0',
                 ):
        self.lr = lr
        self.weight_decay = weight_decay
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.thu_data_loader = thu_data_loader
        self.early_stop = early_stop
        self.epoches = epoches
        self.category_dict = category_dict
        self.loss_weight = loss_weight
        self.use_cuda = use_cuda

        self.emb_dim = emb_dim
        self.mlp_dims = mlp_dims
        self.bert_path = bert_path
        self.dropout = dropout
        self.emb_type = emb_type
        self.model_name = model_name
        self.FuClassifier_path = FuClassifier_path

        if self.use_cuda:
            self.gpu = gpu
        else:
            self.gpu = None

        if not os.path.exists(save_param_dir):
            os.makedirs(save_param_dir)
            self.save_param_dir = save_param_dir
        else:
            self.save_param_dir = save_param_dir

        self.model_save_path = os.path.join(self.save_param_dir + self.model_name + '.pkl')

    def train(self, logger=None):
        if logger:
            logger.info('start training......')
        if self.model_name == "FuzzyMultiDomainFENDModel":
            self.model = FuzzyMultiDomainFENDModel(self.emb_dim, self.mlp_dims, self.bert_path, self.dropout,
                                                   self.emb_type, self.FuClassifier_path, self.gpu)

        if self.use_cuda:
            self.model = self.model.cuda()
        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        val_recorder = Recorder(self.early_stop, os.path.join(self.save_param_dir,
                                                              "val" + self.model_name + '.csv'))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
        for epoch in range(self.epoches):
            self.model.train()
            train_data_iter = tqdm.tqdm(self.train_loader)
            avg_loss = Averager()
            loss = 0

            for step_n, batch in enumerate(train_data_iter):
                batch_data = data2gpu(batch, self.use_cuda)
                label = batch_data['label']
                category = batch_data['category']
                optimizer.zero_grad()
                label_pred = self.model(**batch_data)
                loss = loss_fn(label_pred, label.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                avg_loss.add(loss.item())

            print('Training Epoch {}; Loss {}'.format(epoch + 1,loss))

            results = self.test(self.val_loader)
            mark = val_recorder.add(results)
            if mark == 'save':
                torch.save(self.model.state_dict(), self.model_save_path)
            elif mark == 'esc':
                break
            else:
                continue
        results = self.test(self.test_loader)
        print(results, "\n\n")
        return self.model_save_path

    def test(self, dataloader):
        pred = []
        label = []
        category = []
        avg_loss = Averager()
        loss = 0
        self.model.eval()
        data_iter = tqdm.tqdm(dataloader)
        loss_fn = torch.nn.BCELoss()
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data2gpu(batch, self.use_cuda)
                batch_label = batch_data['label']
                batch_category = batch_data['category']
                batch_label_pred = self.model(**batch_data)

                loss = loss_fn(batch_label_pred, batch_label.float())
                avg_loss.add(loss.item())

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(batch_label_pred.detach().cpu().numpy().tolist())
                category.extend(batch_category.detach().cpu().numpy().tolist())

        return metrics(label, pred, category, self.category_dict, avg_loss.item())

    def final_test(self, loader):
        if self.model_name == "FuzzyMultiDomainFENDModel":
            self.model = FuzzyMultiDomainFENDModel(self.emb_dim, self.mlp_dims, self.bert_path, self.dropout,
                                                   self.emb_type, self.FuClassifier_path, self.gpu)
        self.model.load_state_dict(torch.load(self.model_save_path))
        if self.use_cuda:
            self.model = self.model.cuda()
        return self.test(loader)
