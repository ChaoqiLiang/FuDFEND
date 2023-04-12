from sklearn.metrics import recall_score, precision_score, f1_score, accuracy_score, roc_auc_score
import numpy as np
import pandas as pd
import os


class Recorder():

    def __init__(self, early_step, path):
        self.best = {'loss': float('inf'), '总f1': 0}
        self.cur = {'loss': 0, '总f1': 0}
        self.bestindex = 0
        self.curindex = 0
        self.early_step = early_step
        self.save_path = path
        self.df = pd.DataFrame()

    def add(self, x):
        self.curindex += 1
        self.cur = {"epoch": self.curindex, **x}
        print("curent", self.cur)
        return self.judge()

    def judge(self):
        if self.cur['总f1'] > self.best['总f1'] or (
                self.cur['loss'] < self.best['loss'] and self.cur['总f1'] == self.best['总f1']):
            self.best = self.cur
            self.bestindex = self.curindex
            self.showfinal()
            return 'save'
        self.showfinal()
        if self.curindex - self.bestindex >= self.early_step:
            return 'esc'
        else:
            return 'continue'

    def showfinal(self):
        self.cur = {**self.cur, "总f1_Max": self.best['总f1']}
        self.df = pd.concat([self.df, pd.DataFrame([self.cur])])
        self.df.to_csv(self.save_path, index=False)
        print("Best", self.best)


def metrics(y_true, y_pred, category, category_dict, loss):
    res_by_category = {}
    metrics_by_category = {}
    reverse_category_dict = {}
    for k, v in category_dict.items():
        reverse_category_dict[v] = k + "f1"
        res_by_category[k + "f1"] = {"y_true": [], "y_pred": []}

    for i, c in enumerate(category):
        c = reverse_category_dict[c]
        res_by_category[c]['y_true'].append(y_true[i])
        res_by_category[c]['y_pred'].append(y_pred[i])

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = {
                'auc': roc_auc_score(res['y_true'], res['y_pred']).round(4).tolist()
            }
        except Exception as e:
            metrics_by_category[c] = {
                'auc': 0
            }
    y_pred = np.around(np.array(y_pred)).astype(int)
    metrics_by_category['总f1'] = f1_score(y_true, y_pred, average='macro')

    for c, res in res_by_category.items():
        try:
            metrics_by_category[c] = f1_score(res['y_true'], np.around(np.array(res['y_pred'])).astype(int),
                                              average='macro').round(4).tolist()
        except Exception as e:
            metrics_by_category[c] = 0
            """{
                'precision': 0,
                'recall': 0,
                'fscore': 0,
                'auc': 0,
                'acc': 0
            }"""

    metrics_by_category["loss"] = loss
    return metrics_by_category


def data2gpu(batch, use_cuda):
    if use_cuda:
        batch_data = {
            'content': batch[0].cuda(),
            'content_masks': batch[1].cuda(),
            'label': batch[2].cuda(),
            'category': batch[3].cuda()
        }
    else:
        batch_data = {
            'content': batch[0],
            'content_masks': batch[1],
            'label': batch[2],
            'category': batch[3]
        }
    return batch_data


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v
