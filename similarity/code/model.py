from TD.CosineMimicLoss import CosineMimicLoss
from TD.CosineMimicLoss import myEvaluator
from TD.CosineMimicLoss import get_callback_save_fn

import torch
from sentence_transformers import SentenceTransformer, InputExample, models
from torch.utils.data import DataLoader
from typing import Iterable, Dict
from torch import Tensor
from sentence_transformers.util import batch_to_device
import torch.nn.functional as F
import numpy as np
from datetime import datetime


def loading_training_set_from_file(filepath_lst, num, max_sent_words = 128):

    train_examples = []


    fileContent = []
    for filepath in filepath_lst:

        file = open(filepath)
        fileContent_tmp = file.readlines()
        fileContent += fileContent_tmp
        file.close()

    index_list = list()
    for i in range(0, len(fileContent), 3):
        sentence_1 = fileContent[i+1].strip()
        sentence_2 = fileContent[i+2].strip()
        len1 = len(sentence_1.split(' '))
        len2 = len(sentence_2.split(' '))
        if max(len1, len2) <= max_sent_words:
            index_list.append(i)

    index_list = np.asarray(index_list)
    index_list = np.random.choice(index_list, num)

    for i in index_list:
        lb = int(fileContent[i].strip())
        sentence_1 = fileContent[i+1].strip()
        sentence_2 = fileContent[i+2].strip()
        train_examples.append(InputExample(texts = [sentence_1, sentence_2], label = lb))

    return train_examples


def loading_training_set():

    num = 10000
    data_list = [
        {'filepath': ['../data/training_set_for_label_0'], 'num': num},
        {'filepath': ['../data/training_set_for_label_1.cosine', '../data/training_set_for_label_1.t1', '../data/training_set_for_label_1.t2'], 'num': num},
        {'filepath': ['../data/training_set_for_label_2'], 'num': num},
        {'filepath': ['../data/training_set_for_label_3'], 'num': num}
    ]

    train_examples = []
    for data in data_list:
        filepath, num = data['filepath'], data['num']
        train_examples += loading_training_set_from_file(filepath, num)

    return train_examples


def loading_evaluation_set(flag):

    file = open('../data/evaluation_set.txt')
    fileContent = file.readlines()
    file.close()

    if flag == 'for_train_all_model':

        sentence_1_list = []
        sentence_2_list = []
        label_list = []

        for i in range(0, len(fileContent), 3):
            label = int(fileContent[i].strip())
            sentence_1 = fileContent[i+1].strip()
            sentence_2 = fileContent[i+2].strip()
            sentence_1_list.append(sentence_1)
            sentence_2_list.append(sentence_2)
            label_list.append(label)


        return (sentence_1_list, sentence_2_list, label_list)

    elif flag == 'for_evaluation':

        train_examples = []

        for i in range(0, len(fileContent), 3):
            lb = int(fileContent[i].strip())
            sentence_1 = fileContent[i+1].strip()
            sentence_2 = fileContent[i+2].strip()
            train_examples.append(InputExample(texts = [sentence_1, sentence_2], label = lb))

        return train_examples


def train_all_model():

    model = SentenceTransformer('all-mpnet-base-v2')
    train_loss = CosineMimicLoss(model, feature_dim = model.get_sentence_embedding_dimension())

    train_examples = loading_training_set()
    train_dataloader = DataLoader(train_examples, shuffle = True, batch_size = 16)

    sentence_1_list, sentence_2_list, label_list = loading_evaluation_set('for_train_all_model')
    myevaluator = myEvaluator(sentence_1_list, sentence_2_list, label_list, loss_model = train_loss, batch_size = 16)
    callback_fn = get_callback_save_fn(train_loss, '../data/model_part1.pt')

    train_loss.set_train_all()
    train_loss.set_train_classifier_only()
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=2, warmup_steps=100, optimizer_params={'lr':0.01}, optimizer_class=torch.optim.Adam, weight_decay=1e-5, output_path = '../data/model_part1.pt', evaluator = myevaluator, evaluation_steps = 125, callback = callback_fn)

    evaluation('[only classifier]')

    '''
    torch.cuda.empty_cache()

    train_loss.set_train_all()
    train_loss.reset_train_classifier_only()
    model.train()
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10, warmup_steps=100, optimizer_params={'lr':1e-4}, optimizer_class=torch.optim.SGD, weight_decay=0, output_path = '../data/model_part1.pt', evaluator = myevaluator, evaluation_steps = 125, callback = callback_fn)

    evaluation('[all model]')
    '''


def evaluation(log):

    file = open('../data/evaluation.log', 'a')

    file.write('%s\n\n'%log)

    model = SentenceTransformer('../data/model_part1.pt')
    train_loss = torch.load('../data/model_part2.pt', map_location = model.device)
    train_loss.model = model

    train_examples = loading_evaluation_set('for_evaluation')

    train_dataloader = DataLoader(train_examples, shuffle = False, batch_size = 16)
    train_dataloader.collate_fn = model.smart_batching_collate

    train_loss.set_predict()
    train_loss.eval()

    cnt_all = 0
    cnt_correct = 0

    for data in train_dataloader:
        sentence_batch, label = data
        output = train_loss(sentence_batch, label).detach()
        predict = torch.argmax(output, dim = 1)
        rst = torch.eq(predict, label)
        cnt_all += len(rst)
        cnt_correct += torch.sum(rst).detach().cpu().numpy()

        z = torch.cat([output, torch.unsqueeze(predict, dim = 1), torch.unsqueeze(label, dim = 1)], dim = 1)
        print(z)

        file.write(str(z)+'\n\n')


    print('{:d} correct, {:d} wrong'.format(cnt_correct, cnt_all - cnt_correct))

    file.write('%d correct, %d wrong\n\n\n'%(cnt_correct, cnt_all - cnt_correct))
    file.close()


if __name__ == '__main__':

    train_all_model()

