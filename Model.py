import random
import torch
import torch.nn as nn

import numpy as np
import re
import global_config
import torch.nn.functional as F
from transformers import BartTokenizer
from collections import Counter

from modeling_bart import BartForConditionalGeneration


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """ From fairseq """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss
    smooth_loss = smooth_loss
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss.mean(), nll_loss.mean()


def convert_str_list_to_list(str_list):
    tmp = str_list.strip().replace("[", "").replace("]", "").split(",")
    tmp = [int(i.strip()) for i in tmp if len(i.strip()) > 0]
    return tmp


def Prev_Coreference_Matrix(token_length, src_list, tgt_list):
    coref_matrix = np.zeros([token_length, token_length], dtype=float)
    assert len(src_list) == len(tgt_list)
    for i in range(len(src_list)):
        coref_matrix[src_list[i]][tgt_list[i]] = 1
    for i in range(token_length):
        if sum(coref_matrix[i]) == 0:
            coref_matrix[i][i] = 1
    return coref_matrix


def Adjacent_Coreference_Matrix(token_length, src_list, tgt_list):
    coref_matrix = np.zeros([token_length, token_length], dtype=float)
    assert len(src_list) == len(tgt_list)
    for i in range(len(src_list)):
        coref_matrix[src_list[i]][tgt_list[i]] = 1
        coref_matrix[tgt_list[i]][src_list[i]] = 1
    for i in range(token_length):
        coref_matrix[i][i] = 1
    coref_matrix = coref_matrix / np.sum(coref_matrix, axis=1, keepdims=True)
    return coref_matrix


def All_Coreference_Matrix(token_length, src_list, tgt_list):
    coref_matrix = np.zeros([token_length, token_length], dtype=float)

    set_list = [set()]
    for i in range(len(tgt_list)):
        in_cluster = False
        j = 0
        while (not in_cluster) and (j < len(set_list)):
            if src_list[i] in set_list[j] or tgt_list[i] in set_list[j]:
                in_cluster = True
                set_list[j].add(tgt_list[i])
                set_list[j].add(src_list[i])
            j += 1
        if not in_cluster:
            set_list.append(set())
            set_list[-1].add(tgt_list[i])
            set_list[-1].add(src_list[i])

    cluster_list = [list(i) for i in set_list[1:]]

    for cluster in cluster_list:
        weight = float(1 / (len(cluster)))
        for i in range(len(cluster)):
            for j in range(len(cluster)):
                coref_matrix[cluster[i]][cluster[j]] = weight

    for i in range(token_length):
        if sum(coref_matrix[i]) == 0:
            coref_matrix[i][i] = 1

    return coref_matrix


def build_tensor_with_pre_tokenized_input(batch_input):
    seq_len_list = []
    seq_token_id_list = []
    seq_coref_list = []
    for one in batch_input:
        tmp = one.split("#####")
        seq_token_id_list.append(convert_str_list_to_list(tmp[1]))
        seq_coref_list.append((convert_str_list_to_list(tmp[2]), convert_str_list_to_list(tmp[3])))
        seq_len_list.append(int(tmp[4].strip()))
        assert len(seq_token_id_list[-1]) == seq_len_list[-1]
    max_len = max(seq_len_list)
    seq_token_id_list = [v + [1, ] * (max_len - seq_len_list[k]) for k, v in enumerate(seq_token_id_list)]
    attention_mask = [[1, ] * v + [0, ] * (max_len - v) for k, v in enumerate(seq_len_list)]

    batch_input_tensor = torch.LongTensor(seq_token_id_list).cuda()
    batch_attention_mask = torch.LongTensor(attention_mask).cuda()
    return batch_input_tensor, batch_attention_mask, seq_coref_list


class NeuralSeq2Seq(nn.Module):
    def __init__(self):
        super(NeuralSeq2Seq, self).__init__()

        self.language_backbone = BartForConditionalGeneration.from_pretrained(global_config.pretrained_model, output_hidden_states=False)
        self.tokenizer = BartTokenizer.from_pretrained(global_config.pretrained_tokenizer, use_fast=True)
        print("Loading the pretrained model:", global_config.pretrained_model)

    def supervised_generation(self, batch_input_tensor, batch_attention_mask, batch_decoder_label_tensor, batch_coref_info):
        if global_config.pre_tokenized_samples is True:
            output_logits = self.language_backbone(input_ids=batch_input_tensor, attention_mask=batch_attention_mask, labels=batch_decoder_label_tensor, coref_information=batch_coref_info).logits
        else:
            output_logits = self.language_backbone(input_ids=batch_input_tensor, attention_mask=batch_attention_mask, labels=batch_decoder_label_tensor).logits

        if global_config.using_label_smoothing:
            supervised_loss, _ = label_smoothed_nll_loss(lprobs=F.log_softmax(output_logits, dim=2), target=batch_decoder_label_tensor,
                                                         epsilon=global_config.smooth_epsilon, ignore_index=None)
        else:
            supervised_loss = F.cross_entropy(output_logits.transpose(1, 2), batch_decoder_label_tensor)

        return output_logits, supervised_loss


class Model:
    def __init__(self):
        self.agent = NeuralSeq2Seq()
        if global_config.use_cuda:
            self.agent.cuda()

        self.iter_step = 1

        if global_config.different_learning_rate:
            bert_param_ids = list(map(id, self.agent.language_backbone.parameters()))
            self.backbone_params = filter(lambda p: id(p) in bert_param_ids, self.agent.parameters())
            self.other_params = filter(lambda p: id(p) not in bert_param_ids, self.agent.parameters())
            self.optimizer = torch.optim.AdamW([{'params': self.backbone_params, 'lr': global_config.learning_rate},
                                                {'params': self.other_params, 'lr': 0.001}], lr=global_config.learning_rate)
        else:
            self.optimizer = torch.optim.AdamW(params=self.agent.parameters(), lr=global_config.learning_rate, betas=(0.9, 0.95))

        if global_config.freeze_some_bert_layer:
            for name, param in self.agent.language_backbone.named_parameters():
                layer_num = re.findall("layer\.(\d+)\.", name)
                if len(layer_num) > 0 and int(layer_num[0]) > 2:
                    print("Unfreeze layer:", int(layer_num[0]))
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def adjust_learning_rate(self, backbone_lr, other_lr):
        assert global_config.different_learning_rate
        print("learning rate is changed to:", backbone_lr, other_lr)
        self.optimizer.param_groups[0]["lr"] = backbone_lr
        self.optimizer.param_groups[1]["lr"] = other_lr

    def forward(self, batch, eval_mode=False):
        """ read and process data """
        batch_sample_input_text = [i[0] for i in batch]
        batch_sample_target_text = [i[1] for i in batch]
        batch_coref_list = None

        if global_config.pre_tokenized_samples is True:
            batch_input_tensor, batch_attention_mask, batch_coref_list = build_tensor_with_pre_tokenized_input(batch_sample_input_text)
            graph_attention_heads = []
            for i in range(len(batch_coref_list)):
                graph_attention_heads.append(Adjacent_Coreference_Matrix(batch_input_tensor.size(1), batch_coref_list[i][0], batch_coref_list[i][1]))
            batch_coref_list = torch.Tensor(graph_attention_heads).cuda()
            assert batch_coref_list is not None
        else:
            """ build input tensors """
            batch_encoder_input = self.agent.tokenizer(batch_sample_input_text, return_tensors='pt', padding=True, add_special_tokens=True, truncation=True, max_length=1020)
            batch_input_tensor = batch_encoder_input.data["input_ids"].cuda()
            batch_attention_mask = batch_encoder_input.data["attention_mask"].cuda()

        batch_decoder_label_tensor = self.agent.tokenizer(batch_sample_target_text, return_tensors='pt', padding=True, add_special_tokens=True).data["input_ids"].cuda()

        """ teacher forcing """
        supervised_logits, supervised_loss = self.agent.supervised_generation(batch_input_tensor, batch_attention_mask, batch_decoder_label_tensor, batch_coref_list)

        transferred_sen_text = None
        eval_beam_num = 5
        if eval_mode:
            with torch.no_grad():
                output_sen_token_ids_eval = self.agent.language_backbone.generate(input_ids=batch_input_tensor, attention_mask=batch_attention_mask,
                                                                                  num_beams=eval_beam_num, min_length=global_config.min_gen_len, max_length=global_config.max_gen_len,
                                                                                  early_stopping=False, use_cache=False, coref_info=batch_coref_list)

                transferred_sen_text = [self.agent.tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=True) for i in output_sen_token_ids_eval]

        return supervised_loss, transferred_sen_text

    def infer_forward(self, batch):
        batch_sample_input_text = [i[0] for i in batch]
        batch_coref_list = None

        if global_config.pre_tokenized_samples is True:
            batch_input_tensor, batch_attention_mask, batch_coref_list = build_tensor_with_pre_tokenized_input(batch_sample_input_text)
            graph_attention_heads = []
            for i in range(len(batch_coref_list)):
                graph_attention_heads.append(Adjacent_Coreference_Matrix(batch_input_tensor.size(1), batch_coref_list[i][0], batch_coref_list[i][1]))
            batch_coref_list = torch.Tensor(graph_attention_heads).cuda()
            assert batch_coref_list is not None
        else:
            """ build input tensors """
            encoded_input = self.agent.tokenizer(batch_sample_input_text, return_tensors='pt', padding=True, add_special_tokens=True, truncation=True, max_length=1020)
            batch_input_tensor = encoded_input.data["input_ids"].cuda()
            batch_attention_mask = encoded_input.data["attention_mask"].cuda()

        with torch.no_grad():
            output_sen_token_ids_eval = self.agent.language_backbone.generate(input_ids=batch_input_tensor, attention_mask=batch_attention_mask,
                                                                              num_beams=5, min_length=global_config.min_gen_len, max_length=global_config.max_gen_len,
                                                                              early_stopping=False, use_cache=False, coref_info=batch_coref_list)

            transferred_sen_text = [self.agent.tokenizer.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=False) for i in output_sen_token_ids_eval]

        transferred_sen_text = [i.replace("\n", " ") for i in transferred_sen_text]
        return transferred_sen_text

    def batch_train(self, batch, epoch_number):
        self.agent.train()
        self.optimizer.zero_grad()
        supervised_loss, transferred_sen_text = self.forward(batch)
        supervised_loss.backward()
        self.optimizer.step()

        self.iter_step += 1

        return supervised_loss.item(), transferred_sen_text

    def batch_eval(self, batch):
        self.agent.eval()
        supervised_loss, transferred_sen_text = self.forward(batch, eval_mode=True)
        return supervised_loss.item(), transferred_sen_text

    def batch_infer(self, batch):
        self.agent.eval()
        transferred_sen_text = self.infer_forward(batch)
        return transferred_sen_text

    def save_model(self, save_path):
        """ save model """
        print("Saving model to:", save_path)
        torch.save(self.agent.state_dict(), save_path)

    def load_model(self, load_path):
        """ save model """
        print("Loading model from:", load_path)
        self.agent.load_state_dict(torch.load(load_path))
