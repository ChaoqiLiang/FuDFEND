from transformers import BertTokenizer
import torch
from transformers import BertModel
import torch.nn as nn
import json
import pickle
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import torch
import numpy as np
import random

seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class fuzzy_classifier(torch.nn.Module):
    def __init__(self, bert_path, category_num=9, dropout=0.3, hidden_size=256):
        super(fuzzy_classifier, self).__init__()
        f = open(bert_path + "/config.json", "r")
        config = json.load(f)
        f.close()
        self.bert_dim = config["hidden_size"]
        self.gru = nn.GRU(self.bert_dim, hidden_size)
        self.linear = nn.Linear(hidden_size, category_num)

    def forward(self, X):
        features = self.gru(X.transpose(0, 1))
        outputs = self.linear(features[1])
        return outputs[0]


def word2input(texts, vocab_file, max_len):
    tokenizer = BertTokenizer(vocab_file=vocab_file)
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks


def _init_fn(worker_id):
    np.random.seed(2022)


def read_pkl(path):
    with open(path, "rb") as f:
        t = pickle.load(f)
    return t


def df_filter(df_data):
    df_data = df_data[df_data['category'] != '无法确定']
    return df_data


class bert_data():
    def __init__(self, max_len, batch_size, vocab_file, category_dict, num_workers=0):
        self.max_len = max_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.vocab_file = vocab_file
        self.category_dict = category_dict

    def load_data(self, path, shuffle, drop_last=False):
        self.data = df_filter(read_pkl(path))
        content = self.data['content'].to_numpy()
        label = torch.tensor(self.data['label'].astype(int).to_numpy())
        category = torch.tensor(self.data['category'].apply(lambda c: self.category_dict[c]).to_numpy())
        content_token_ids, content_masks = word2input(content, self.vocab_file, self.max_len)
        dataset = TensorDataset(content_token_ids,
                                content_masks,
                                category
                                )
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=shuffle,
            worker_init_fn=_init_fn,
            drop_last=drop_last
        )
        return dataloader


class Recorder():

    def __init__(self, early_step):
        self.max = {'metric': 0}
        self.cur = {'metric': 0}
        self.maxindex = 0
        self.curindex = 0
        self.early_step = early_step

    def add(self, x):
        self.cur = x
        self.curindex += 1
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['metric'] > self.max['metric']:
            self.max = self.cur
            self.maxindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.maxindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        print("Max", self.max)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def metrics(y_true, y_pred, category, category_dict):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k
        res_by_category[k] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    metrics_by_category['metric'] = (np.array(y_true) == np.array(y_pred)).sum() / len(y_pred)

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'metric': (np.array(res['y_true']) == np.array(res['y_pred'])).sum() / len(res['y_pred'])
            }
        except Exception as e:
            metrics_by_category[c] = {
                'metric': 0
            }
    return metrics_by_category


category_dict = {
    "科技": 0,
    "军事": 1,
    "教育考试": 2,
    "灾难事故": 3,
    "政治": 4,
    "医药健康": 5,
    "财经商业": 6,
    "文体娱乐": 7,
    "社会生活": 8
}
loader = bert_data(170, 32, "pretrained_model/chinese_roberta_wwm_base_ext_pytorch/vocab.txt", category_dict)
train_loader = loader.load_data("data/membership_train_set.pkl", True)
test_loader = loader.load_data("data/membership_test_set.pkl", False)

bert = BertModel.from_pretrained("pretrained_model/chinese_roberta_wwm_base_ext_pytorch").cuda(0).requires_grad_(False)
net = fuzzy_classifier("pretrained_model/chinese_roberta_wwm_base_ext_pytorch").cuda(0)


def test(net, test_loader):
    preds = []
    trues = []
    net.eval()
    test_data_iter = tqdm.tqdm(test_loader)
    for step_n, batch in enumerate(test_data_iter):
        X = batch[0].cuda()
        masks = batch[1].cuda()
        trues += list(batch[2].long().numpy())
        features = bert(X, masks)[0]
        outputs = net(features)
        preds += list(outputs.cpu().argmax(-1).numpy())
    return metrics(trues, preds, trues, category_dict)


max_acc = 0
net.train()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=net.parameters(), lr=0.0005, weight_decay=5e-5)
recorder = Recorder(10)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.98)
for epoch in range(15):
    net.train()
    avg_loss = Averager()
    train_data_iter = tqdm.tqdm(train_loader)
    for step_n, batch in enumerate(train_data_iter):
        X = batch[0].cuda()
        masks = batch[1].cuda()
        labels = batch[2].long().cuda()
        features = bert(X, masks)[0]
        outputs = net(features)
        loss = loss_fn(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        avg_loss.add(loss.item())
    print('Training Epoch {}; Loss {}; '.format(epoch + 1, loss.item()))
    acc = test(net, test_loader)
    mark = recorder.add(acc)
    torch.save(net.state_dict(), 'pretrained_model/membership_function/' + str(epoch+1) + "_" + 'parameter_membership_function.pkl')
