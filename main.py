import os
import random
import time
import re
from collections import Counter
from tqdm import tqdm
import pickle

import torch
import numpy as np
import rouge

from Model import Model
import global_config

os.environ['CUDA_VISIBLE_DEVICES'] = global_config.gpu_id
running_random_number = random.randint(1000, 9999)
print("running_random_number", running_random_number, "\n")

global_rouge_scorer = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                                  max_n=4,
                                  limit_length=True,
                                  length_limit=500,
                                  length_limit_type='words',
                                  apply_avg=True,
                                  apply_best=False,
                                  alpha=0.5,
                                  weight_factor=1.2,
                                  stemming=True)


def prepare_results(metric, p, r, f):
    return '\t{}:\t {:5.2f}\t {:5.2f}\t {:5.2f}'.format(metric, 100.0 * p, 100.0 * r, 100.0 * f)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_batches(data, batch_size):
    batches = []
    for i in range(len(data) // batch_size + bool(len(data) % batch_size)):
        batches.append(data[i * batch_size:(i + 1) * batch_size])

    return batches


def train_process(model, train_data, valid_data, test_data):
    train_epoch = global_config.start_from_epoch

    best_score = {"epoch": 0, "all_loss": 0}
    running_log_name = None

    while train_epoch < global_config.num_epochs:
        print("\n********* Epoch {} ***********".format(train_epoch))
        summary_steps = 0

        random.seed(100 + train_epoch)
        random.shuffle(train_data)

        train_batches = get_batches(train_data, global_config.batch_size)

        if global_config.scheduling_learning_rate:
            if train_epoch < 1:
                print("Running with warm up learning rate.")
                model.adjust_learning_rate(backbone_lr=0.00001, other_lr=0.003)
            else:
                lr_decay = 0.98 ** (train_epoch - 1)
                model.adjust_learning_rate(backbone_lr=0.00002 * lr_decay, other_lr=0.001 * lr_decay)

        for batch in train_batches:
            supervised_loss, transferred_sen_text = model.batch_train(batch, train_epoch)
            summary_steps += 1
            if summary_steps % global_config.batch_loss_print_interval == 0:
                print(train_epoch, summary_steps, "supervised loss", supervised_loss)

        best_score, current_eval_score = evaluate_process(model, valid_data, train_epoch, best_score)
        # best_score, current_eval_score = evaluate_process(model, test_data, train_epoch, best_score)

        if running_log_name is None:
            running_log_name = "./running_log/" + time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()) + "_" + str(running_random_number) + ".txt"
            open(running_log_name, "w").write("Corpus mode: " + global_config.corpus_mode + "\n")

        if running_log_name:
            with open(running_log_name, "a") as fp:
                fp.write("\nEpoch: " + str(train_epoch) + " Rouge Score: " + str(current_eval_score) + "\n")

        train_epoch += 1


def evaluate_process(model, data_test_collection, train_epoch, best_score=None, preview=False, fast_infer=False):
    test_batches = get_batches(data_test_collection, global_config.batch_size)

    all_transferred_sentences, all_gold_sentences = [], []
    all_test_loss = []

    for batch in tqdm(test_batches):
        if fast_infer is False:
            supervised_loss, transferred_sen_text = model.batch_eval(batch)
            all_test_loss.append(supervised_loss)
        else:
            transferred_sen_text = model.batch_infer(batch)
            all_test_loss.append(-1)

        all_transferred_sentences.extend(transferred_sen_text)
        all_gold_sentences.extend([i[1] for i in batch])

    if preview:
        pass

    if global_config.print_all_predictions:
        with open("all_generation.txt", "w", encoding="utf-8") as fp:
            for i in all_transferred_sentences:
                fp.write(i.replace("\n", " ").strip() + "\n")

    print("\nTest Result in epoch:", train_epoch)

    all_gold_sentences = [i.lower() for i in all_gold_sentences]
    all_transferred_sentences = [i.lower() for i in all_transferred_sentences]

    eval_rouge_score = global_rouge_scorer.get_scores(references=all_gold_sentences, hypothesis=all_transferred_sentences)
    rouge_res = ""
    for metric, results in sorted(eval_rouge_score.items(), key=lambda x: x[0]):
        if metric in ["rouge-1", "rouge-2", "rouge-l"]:
            print(prepare_results(metric, results['p'], results['r'], results['f']))
            rouge_res = rouge_res + '{:5.2f}'.format(100 * results['f']) + "-"

    print("ROUGE 1-2-L F:", rouge_res, "\n")
    current_eval_score = eval_rouge_score["rouge-1"]["f"]

    if any(all_test_loss):
        print("Test loss:", np.mean(all_test_loss))

    if best_score:
        if current_eval_score > best_score["all_loss"]:
            best_score = {"epoch": train_epoch, "all_loss": current_eval_score}

    if global_config.save_model and global_config.train and train_epoch >= 1:
        model.save_model("./saved_models/best_model_" + str(running_random_number) + "_epoch" + str(train_epoch) + "_" + str(current_eval_score)[:7] + ".pth")

    current_print_score_str = "ROUGE 1-2-L F:" + str(rouge_res)

    return best_score, current_print_score_str


if __name__ == '__main__':
    setup_seed(100)

    if global_config.train:
        print("Reading train data...")
        train_file_prefix = global_config.data_path + "train."
        train_sample_list = list(zip(open(train_file_prefix + "source", encoding="utf-8").readlines(), open(train_file_prefix + "target", encoding="utf-8").readlines()))
        train_sample_list = [[i[0].strip(), i[1].strip()] for i in train_sample_list]

        data_train = train_sample_list
        print('Train Dataset size: %d' % (len(data_train)))

        print("Reading validation data...")
        val_file_prefix = global_config.data_path + "val."
        val_sample_list = list(zip(open(val_file_prefix + "source", encoding="utf-8").readlines(), open(val_file_prefix + "target", encoding="utf-8").readlines()))
        val_sample_list = [[i[0].strip(), i[1].strip()] for i in val_sample_list]

        data_valid = val_sample_list
        print('Validation Dataset size: %d' % (len(data_valid)))

    print("Reading test data...")
    test_file_prefix = global_config.data_path + "test."
    test_sample_list = list(zip(open(test_file_prefix + "source", encoding="utf-8").readlines(), open(test_file_prefix + "target", encoding="utf-8").readlines()))
    test_sample_list = [[i[0].strip(), i[1].strip()] for i in test_sample_list]

    data_test = test_sample_list
    print('Test Dataset size: %d' % (len(data_test)))

    model = Model()

    train_mode = global_config.train

    if train_mode:
        if global_config.start_from_epoch != 0:
            load_model_path = global_config.load_model_path
            model.load_model(load_model_path)

        train_process(model, data_train, data_valid, data_test)
    else:
        load_model_path = global_config.load_model_path
        model.load_model(load_model_path)
        evaluate_process(model, data_test, -999, fast_infer=True)
