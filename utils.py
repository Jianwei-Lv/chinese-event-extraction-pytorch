# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import json
from const import TRIGGERS,ENTITIES,ARGUMENTS
PAD, CLS ,NONE= '[PAD]', '[CLS]' ,'NONE' # padding符号, bert中综合信息符号

def build_vocab(labels_trigger,labels_entities, BIO_tagging=True):
    all_labels = [NONE]
    for label in labels_trigger:
        if BIO_tagging:
            all_labels.append('B-T-{}'.format(label))
            all_labels.append('I-T-{}'.format(label))
        else:
            all_labels.append(label)
    for label in labels_entities:
        if BIO_tagging:
            all_labels.append('B-E-{}'.format(label))
            all_labels.append('I-E-{}'.format(label))
        else:
            all_labels.append(label)
    label2idx = {tag: idx for idx, tag in enumerate(all_labels)}
    idx2label = {idx: tag for idx, tag in enumerate(all_labels)}

    return all_labels, label2idx, idx2label

all_triggers_entities, trigger_entities2idx, idx2trigger_entities = build_vocab(TRIGGERS,ENTITIES)
# all_entities, entity2idx, idx2entity = build_vocab(ENTITIES)
# all_postags, postag2idx, idx2postag = build_vocab(POSTAGS, BIO_tagging=False)
all_arguments, argument2idx, idx2argument = build_vocab(ARGUMENTS,[], BIO_tagging=False)

def build_dataset(config):

    def load_dataset(path, pad_size=128):
        cut_off=pad_size
        contents = []

        with open(path, 'r', encoding='UTF-8') as f:
            data = json.load(f)
            for item in data:

                words=[ item['sentence'][i] for i in range(len(item['sentence']))]
                token=[]
                for w in words:
                    t = config.tokenizer.tokenize(w)
                    token.extend(t)


                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        mask = [1] * pad_size
                        token_ids = token_ids[:pad_size]
                        seq_len = pad_size
                triggers_entities=[NONE for _ in range(len(token))][:cut_off]
                arguments = {
                    'candidates': [
                        # ex. (5, 6, "entity_type_str"), ...
                    ],
                    'events': {
                        # ex. (1, 3, "trigger_type_str"): [(5, 6, "argument_role_idx"), ...]
                    },
                }
                try:
                    for entity_mention in item['arguments']:
                        start = entity_mention['start']
                        if start >= cut_off:
                            continue
                        end = min(entity_mention["end"], cut_off)
                        arguments['candidates'].append((start+1, end+1, entity_mention['entity_type']))

                        for i in range(start, end):
                            entity_type = entity_mention['entity_type']
                            if i == start:
                                entity_type = 'B-E-{}'.format(entity_type)
                            else:
                                entity_type = 'I-E-{}'.format(entity_type)

                            triggers_entities[i+1] = entity_type


                    for event_mention in item['trigger']:
                        if event_mention['start'] >= cut_off:
                            continue
                        for i in range(event_mention['start'],min(event_mention['end'], cut_off)):
                            trigger_type = event_mention['event_type']
                            if i == event_mention['start']:

                                triggers_entities[i+1]= 'B-T-{}'.format(trigger_type)
                            else:

                                triggers_entities[i+1] = 'I-T-{}'.format(trigger_type)

                        event_key = (event_mention['start']+1, min(event_mention['end'], cut_off)+1,event_mention['event_type'])
                        arguments['events'][event_key] = []
                        for argument in item['arguments']:
                            if argument['start'] >= cut_off:
                                continue
                            role = argument['role']

                            arguments['events'][event_key].append(
                                (argument['start']+1, min(argument['end'], cut_off)+1, argument2idx[role]))

                    triggers_entities_ids=[trigger_entities2idx[i] for i in triggers_entities]
                    if pad_size:
                        if len(triggers_entities_ids) < pad_size:

                            triggers_entities_ids += ([0] * (pad_size - len(triggers_entities_ids)))
                        else:

                            triggers_entities_ids = triggers_entities_ids[:pad_size]


                    contents.append((token_ids,triggers_entities_ids,seq_len,mask,token,triggers_entities,arguments))
                except:

                    continue

        return contents
    train = load_dataset(config.train_path, config.pad_size)

    return train


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        # mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        mask = [_[3] for _ in datas]
        words=[_[4] for _ in datas]
        trigger_entities = [_[5] for _ in datas]
        arguments=[_[-1] for _ in datas]


        return (x, seq_len, mask,words,trigger_entities,arguments), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def find_triggers(labels):
    """
    :param labels: ['B-Conflict:Attack', 'I-Conflict:Attack', 'O', 'B-Life:Marry']
    :return: [(0, 2, 'Conflict:Attack'), (3, 4, 'Life:Marry')]
    """
    result_trigger = []
    result_entities=[]
    labels = [label.split('-') for label in labels]

    for i in range(len(labels)):
        if labels[i][0] == 'B':
            if labels[i][1]=='T':
                result_trigger.append([i, i + 1, labels[i][2]])
            elif labels[i][1]=='E':
                result_entities.append([i, i + 1, labels[i][2]])

    for item in result_trigger:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I' and labels[j][1]=='T':
                j = j + 1
                item[1] = j
            else:
                break
    for item in result_entities:
        j = item[1]
        while j < len(labels):
            if labels[j][0] == 'I' and labels[j][1]=='E':
                j = j + 1
                item[1] = j
            else:
                break

    return [tuple(item) for item in result_trigger],[tuple(item) for item in result_entities]


def calc_metric(y_true, y_pred):
    """
    :param y_true: [(tuple), ...]
    :param y_pred: [(tuple), ...]
    :return:
    """
    num_proposed = len(y_pred)
    num_gold = len(y_true)

    y_true_set = set(y_true)
    num_correct = 0
    for item in y_pred:
        if item in y_true_set:
            num_correct += 1

    print('proposed: {}\tcorrect: {}\tgold: {}'.format(num_proposed, num_correct, num_gold))

    if num_proposed != 0:
        precision = num_correct / num_proposed
    else:
        precision = 1.0

    if num_gold != 0:
        recall = num_correct / num_gold
    else:
        recall = 1.0

    if precision + recall != 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0

    return precision, recall, f1
