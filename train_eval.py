# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss
# from sklearn import metrics
import time,os
from utils import get_time_dif,calc_metric
from pytorch_pretrained_bert.optimization import BertAdam
from utils import all_triggers_entities, trigger_entities2idx, idx2trigger_entities,find_triggers,all_arguments, argument2idx, idx2argument


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def eval(model, iterator, fname):
    model.eval()

    words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all = [], [], [], [], []
    with torch.no_grad():
        # for i, batch in enumerate(iterator):
        for i, (test, labels) in enumerate(iterator):
            trigger_logits, trigger_entities_hat_2d, triggers_y_2d, argument_hidden_logits, arguments_y_1d, argument_hidden_hat_1d, argument_hat_2d, argument_keys = model(test, labels)


            words_all.extend(test[3])
            triggers_all.extend(test[4])
            triggers_hat_all.extend(trigger_entities_hat_2d.cpu().numpy().tolist())
            arguments_2d=test[-1]
            arguments_all.extend(arguments_2d)
            if len(argument_keys) > 0:
                arguments_hat_all.extend(argument_hat_2d)
            else:
                batch_size = len(arguments_2d)
                argument_hat_2d = [{'events': {}} for _ in range(batch_size)]
                arguments_hat_all.extend(argument_hat_2d)

    triggers_true, triggers_pred, arguments_true, arguments_pred = [], [], [], []
    with open('temp', 'w',encoding='utf-8') as fout:
        for i, (words, triggers, triggers_hat, arguments, arguments_hat) in enumerate(zip(words_all, triggers_all, triggers_hat_all, arguments_all, arguments_hat_all)):
            triggers_hat = triggers_hat[:len(words)]
            triggers_hat = [idx2trigger_entities[hat] for hat in triggers_hat]

            # [(ith sentence, t_start, t_end, t_type_str)]
            triggers_true_,entities_true=find_triggers(triggers[:len(words)])
            triggers_pred_, entities_pred = find_triggers(triggers_hat)
            triggers_true.extend([(i, *item) for item in triggers_true_])
            triggers_pred.extend([(i, *item) for item in triggers_pred_])

            # [(ith sentence, t_start, t_end, t_type_str, a_start, a_end, a_type_idx)]
            for trigger in arguments['events']:
                t_start, t_end, t_type_str = trigger
                for argument in arguments['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    arguments_true.append(( t_type_str, a_start, a_end, a_type_idx))

            for trigger in arguments_hat['events']:
                t_start, t_end, t_type_str = trigger
                if t_start>=len(words) or t_end>=len(words):
                    continue
                for argument in arguments_hat['events'][trigger]:
                    a_start, a_end, a_type_idx = argument
                    if a_start >= len(words) or a_end >= len(words):
                        continue
                    arguments_pred.append((t_type_str, a_start, a_end, a_type_idx))

            for w, t, t_h in zip(words, triggers, triggers_hat):
                fout.write('{}\t{}\t{}\n'.format(w, t, t_h))
            fout.write('#arguments#{}\n'.format(arguments['events']))
            fout.write('#arguments_hat#{}\n'.format(arguments_hat['events']))
            fout.write("\n")

    # print(classification_report([idx2trigger[idx] for idx in y_true], [idx2trigger[idx] for idx in y_pred]))

    print('[trigger classification]')
    trigger_p, trigger_r, trigger_f1 = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p, trigger_r, trigger_f1))

    print('[argument classification]')
    argument_p, argument_r, argument_f1 = calc_metric(arguments_true, arguments_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p, argument_r, argument_f1))
    print('[trigger identification]')
    triggers_true = [(item[0], item[1], item[2]) for item in triggers_true]
    triggers_pred = [(item[0], item[1], item[2]) for item in triggers_pred]
    trigger_p_, trigger_r_, trigger_f1_ = calc_metric(triggers_true, triggers_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(trigger_p_, trigger_r_, trigger_f1_))

    print('[argument identification]')
    arguments_true = [(item[0], item[1], item[2]) for item in arguments_true]
    arguments_pred = [(item[0], item[1], item[2]) for item in arguments_pred]
    argument_p_, argument_r_, argument_f1_ = calc_metric(arguments_true, arguments_pred)
    print('P={:.3f}\tR={:.3f}\tF1={:.3f}'.format(argument_p_, argument_r_, argument_f1_))

    metric = '[trigger classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p, trigger_r, trigger_f1)
    metric += '[argument classification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p, argument_r, argument_f1)
    metric += '[trigger identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(trigger_p_, trigger_r_, trigger_f1_)
    metric += '[argument identification]\tP={:.3f}\tR={:.3f}\tF1={:.3f}\n'.format(argument_p_, argument_r_, argument_f1_)
    final = fname + ".P%.2f_R%.2f_F%.2f" % (trigger_p, trigger_r, trigger_f1)
    with open(final, 'w',encoding='utf-8') as fout:
        result = open("temp", "r",encoding='utf-8').read()
        fout.write("{}\n".format(result))
        fout.write(metric)
    os.remove("temp")
    return metric,trigger_f1,argument_f1


def train(config, model, train_iter, test_iter):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    model.train()
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # optimizer = BertAdam(optimizer_grouped_parameters,
    #                      lr=config.learning_rate,
    #                      warmup=0.05,
    #                      t_total=len(train_iter) * config.num_epochs)
    trigger_F1=0
    argument_F1=0

    for epoch in range(config.num_epochs):
        model.train()
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (trains, labels) in enumerate(train_iter):
            model.zero_grad()
            trigger_loss, trigger_entities_hat_2d, triggers_y_2d,argument_hidden_logits, arguments_y_1d, argument_hidden_hat_1d, argument_hat_2d,argument_keys= model(trains,labels)


            # trigger_logits = trigger_logits.view(-1, trigger_logits.shape[-1])
            # trigger_loss = criterion(trigger_logits, triggers_y_2d.view(-1))
            if len(argument_keys)>0 :
                argument_loss = criterion(argument_hidden_logits, arguments_y_1d)

                loss = trigger_loss + argument_loss
            else:
                loss=trigger_loss
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            loss.backward()

            optimizer.step()
            # if i % 100 == 0:  # monitoring
            print("step: {}, loss: {}".format(i, loss.item()))


        print(f"=========eval test at epoch={epoch}=========")
        metric_test, trigger_f1, argument_f1 = eval(model, test_iter, 'nanhai_data/test_result/'+str(epoch) + '_test')


        if trigger_F1 < trigger_f1:
            trigger_F1 = trigger_f1
            torch.save(model, "latest_model_2.pt")
        if argument_F1 < argument_f1:
            argument_F1 = argument_f1
            torch.save(model, "argument_latest_model_2.pt")
        print('best trigger F1:')
        print(trigger_F1)
        print('best argument F1:')
        print(argument_F1)

