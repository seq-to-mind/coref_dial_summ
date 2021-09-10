import copy
from tqdm import tqdm
import re
import numpy as np
import pickle
from transformers import AutoTokenizer


def Prev_Coreference_Matrix(token_length, src_list, tgt_list):
    """ build the prev-linked coreference matrix """
    coref_matrix = np.zeros([token_length, token_length], dtype=float)
    assert len(src_list) == len(tgt_list)
    for i in range(len(src_list)):
        coref_matrix[src_list[i]][tgt_list[i]] = 1
    for i in range(token_length):
        if sum(coref_matrix[i]) == 0:
            coref_matrix[i][i] = 1
    return coref_matrix


class BuildSampleWithCoreferenceInfo:
    def __init__(self):
        """ Here we use the tokenizer from BART """
        self.global_tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")

    def build_sample_with_coref_to_file(self, task_name, input_list, aux_condition_name_file=None, conditional_file_path=None, debug=False):
        """
        :param task_name: indicate the task type: train/val/test
        :param aux_condition_name_file: each row will contain the speaker roles / personal named entities
        :param input_list: the list of conversations
        :param conditional_file_path: the list of conversations with conditional planning
        :param debug: for debug print
        :return: directly write a file.
        """

        if conditional_file_path is not None:
            conditional_line_list = open(conditional_file_path, encoding="utf-8").readlines()
            assert len(conditional_line_list) == len(input_list)

        output_fp = open(task_name + ".source", "w", encoding="utf-8")

        if aux_condition_name_file is not None:
            aux_name_list = open(aux_condition_name_file, encoding="utf-8").readlines()
            assert len(aux_name_list) == len(input_list)
        else:
            aux_name_list = None

        tmp_line_idx = 0
        for tmp_k, tmp_dict_node in tqdm(enumerate(input_list)):
            """ we use the multiple coreference resolution outputs """
            for coref_idx, coref_type in enumerate(['newline', 'dot', 'sharp', 'newline']):
                tmp_i = tmp_dict_node[coref_type]
                tmp_token_list = tmp_i[1]["document"]
                tmp_clusters = tmp_i[1]["clusters"]

                # print(tmp_i[0])
                raw_coref_cluster_info = []
                for k, i in enumerate(tmp_clusters):
                    one_list = [" ".join(tmp_token_list[j[0]:j[1] + 1]) for j in i]
                    raw_coref_cluster_info.append((k, one_list))

                """ Tackle the issue that some speaker names are not included in coreference chains """
                tmp_new_coref_cluster_info = copy.deepcopy(raw_coref_cluster_info)

                tmp_titled_speakers = set([i[:-1] for i in tmp_i[0].split() if i[-1] == ":" and i.istitle()])
                tmp_speaker_label_dict = {}
                for k, v in enumerate(tmp_titled_speakers):
                    tmp_cluster_res = [j[0] for j in tmp_new_coref_cluster_info if v in j[1]]
                    if len(tmp_cluster_res) < 1 and v not in tmp_speaker_label_dict.keys():
                        tmp_speaker_label_dict[v] = len(tmp_new_coref_cluster_info) + 30
                        tmp_new_coref_cluster_info.append((len(tmp_new_coref_cluster_info) + 30, [v]))
                    else:
                        if len(tmp_cluster_res) == 1:
                            tmp_speaker_label_dict[v] = tmp_cluster_res[0]
                        if len(tmp_cluster_res) > 1:
                            """ Here we select the first found token as the cluster label """
                            q_list = [(q, tmp_clusters[q][tmp_new_coref_cluster_info[q][1].index(v)][0]) for q in tmp_cluster_res]
                            q_list = sorted(q_list, key=lambda x: x[1])
                            tmp_speaker_label_dict[v] = q_list[0][0]

                if aux_name_list is not None:
                    assert len(re.findall("\}\s+\#", aux_name_list[tmp_k])) == 1
                    aux_one_cond_name_set = set(re.sub("[\#\.\|\{\}]]", " ", aux_name_list[tmp_k].split("}")[0]).split())
                else:
                    aux_one_cond_name_set = set()

                continue_flag = False
                for tmp_item in tmp_new_coref_cluster_info:
                    tmp_small_set = set([i.strip().split()[0] for i in tmp_item[1]])
                    intersection = tmp_small_set & (set(tmp_titled_speakers) | set(aux_one_cond_name_set))
                    if len(intersection) > 1:
                        if len(set([i[:2] for i in intersection])) > 1:
                            if coref_idx == 3:
                                print("one plausible coreference chain.")
                            continue_flag = True

                if continue_flag is False:
                    break

            """ Further add titled words to increase coverage """
            tmp_titled_other_tokens = set([i for i in tmp_i[1]["document"] if len(i) > 2 and i.istitle()])
            for k, v in enumerate(tmp_titled_other_tokens):
                tmp_cluster_res = [j[0] for j in tmp_new_coref_cluster_info if v in j[1]]
                if len(tmp_cluster_res) < 1 and v not in tmp_speaker_label_dict.keys():
                    tmp_cluster_res = [(j[0], j[1].count(v)) for j in tmp_new_coref_cluster_info if v in " ".join(j[1]).split()]
                    tmp_cluster_res = sorted(tmp_cluster_res, key=lambda x: x[1], reverse=True)
                    if len(tmp_cluster_res) > 0:
                        tmp_speaker_label_dict[v] = tmp_cluster_res[0][0]
                    else:
                        tmp_speaker_label_dict[v] = len(tmp_new_coref_cluster_info) + 100
                        tmp_new_coref_cluster_info.append((len(tmp_new_coref_cluster_info) + 100, [v]))

            """ Adding spaces in tokenized list, to recover the same tokenization via BART """
            tmp_doc = copy.deepcopy(" " + tmp_i[0])

            tmp_token_list_with_space = []
            for k, v in enumerate(tmp_token_list):
                find_idx = str(tmp_doc).index(v)
                if find_idx > 0 and tmp_doc[find_idx - 1] == " ":
                    tmp_token_list_with_space.append([" " + v, -1])
                else:
                    tmp_token_list_with_space.append([v, -1])
                tmp_doc = tmp_doc[find_idx + len(v):]

            """ Labeling the token list with the coreference cluster labels """
            """ From the longer spans to shorter spans, to avoiding labels to be re-changed """
            tmp_span_len_list = []
            for i in tmp_clusters:
                tmp_span_len_list.extend([j[-1] + 1 - j[0] for j in i])

            tmp_span_len_list = list(set(tmp_span_len_list))
            tmp_span_len_list = sorted(tmp_span_len_list, reverse=True)

            for one_len in tmp_span_len_list:
                for i in range(len(tmp_clusters)):
                    for j in tmp_clusters[i]:
                        if (j[1] + 1 - j[0]) == one_len:
                            for e in j:
                                tmp_token_list_with_space[e][1] = i

            """ Tackle the issue that speaker names do not have coreference """
            assert len(tmp_token_list_with_space) == len(tmp_token_list)
            for k in range(len(tmp_token_list_with_space)):
                if (k == len(tmp_token_list_with_space) - 1 or tmp_token_list_with_space[k + 1][0].strip() == ":") \
                        and tmp_token_list_with_space[k][0].strip() in tmp_speaker_label_dict.keys() \
                        and tmp_token_list_with_space[k][1] == -1:
                    tmp_token_list_with_space[k][1] = tmp_speaker_label_dict[tmp_token_list_with_space[k][0].strip()]
                    # print(tmp_token_list_with_space)

            """ Merge the token list with the same coreference cluster """
            merged_tmp_token_list_with_space = []
            current_merge_set = []
            current_cluster_id_to_merge = -999
            for i in range(len(tmp_token_list_with_space)):
                if tmp_token_list_with_space[i][1] == current_cluster_id_to_merge:
                    current_merge_set.append(tmp_token_list_with_space[i])
                    current_cluster_id_to_merge = tmp_token_list_with_space[i][1]
                else:
                    if len(current_merge_set) > 0:
                        merged_tmp_token_list_with_space.append([[j[0] for j in current_merge_set], current_merge_set[0][1]])
                        current_merge_set = []
                    current_merge_set.append(tmp_token_list_with_space[i])
                    current_cluster_id_to_merge = tmp_token_list_with_space[i][1]
                if i == len(tmp_token_list_with_space) - 1 and len(current_merge_set) > 0:
                    merged_tmp_token_list_with_space.append([[j[0] for j in current_merge_set], current_merge_set[0][1]])
                    current_merge_set = []

            # print(merged_tmp_token_list_with_space)

            """ Using the BART tokenizer to process the new token list """
            for i in range(len(merged_tmp_token_list_with_space)):
                merged_tmp_token_list_with_space[i][0] = self.global_tokenizer.tokenize("".join(merged_tmp_token_list_with_space[i][0]))

            """ V2 only point to the first token of spans """
            tmp_token_list_with_cluster_ids = []
            for i in merged_tmp_token_list_with_space:
                for j in range(len(i[0])):
                    if j == 0:
                        tmp_token_list_with_cluster_ids.append([i[0][j], i[1]])
                    else:
                        tmp_token_list_with_cluster_ids.append([i[0][j], -1])

            if debug:
                tmp_t = " ".join(self.global_tokenizer.tokenize(tmp_i[0])).strip()
                tmp_c = " ".join([j[0] for j in tmp_token_list_with_cluster_ids]).strip()
                print("\n", tmp_t, "\n", tmp_c)

            """ Adding coreference of the conditional personal names """
            if conditional_file_path is not None:
                assert len(re.findall("\}\s+\#", conditional_line_list[tmp_k])) == 1
                conditional_names = (conditional_line_list[tmp_k].split("}")[0] + "} #").split()
                conditional_names = [[i.strip(), -1] for i in conditional_names]

                for k, v in enumerate(conditional_names):
                    if v[0] not in ["{", "}", "#", "|"]:
                        tmp_cluster_res = [(j[0], j[1].count(v[0])) for j in tmp_new_coref_cluster_info if v[0] in j[1]]
                        if len(tmp_cluster_res) > 0:
                            conditional_names[k][1] = tmp_cluster_res[0][0]
                            # if len(tmp_cluster_res) > 1:
                            #     print(tmp_cluster_res)
                        else:
                            """ splitting every name in the cluster keys, then find more names """
                            tmp_cluster_res = [(j[0], j[1].count(v[0])) for j in tmp_new_coref_cluster_info if v[0] in " ".join(j[1]).split()]

                            if len(tmp_cluster_res) > 0:
                                conditional_names[k][1] = tmp_cluster_res[0][0]
                            else:
                                """ To tackle the exception of names are not included """
                                tmp_new_coref_cluster_info.append((len(tmp_new_coref_cluster_info) + 50, [v[0]]))
                                for n, i in enumerate(tmp_token_list_with_cluster_ids):
                                    if i[0][1:] == v[0] and i[1] == -1:
                                        tmp_token_list_with_cluster_ids[n][1] = tmp_new_coref_cluster_info[-1][0]
                                        conditional_names[k][1] = tmp_new_coref_cluster_info[-1][0]
                                print("\n\n\n")
                                print(tmp_i[0])
                                print(v[0])
                                print(tmp_token_list_with_cluster_ids)
                                pass

                print(conditional_names)
                condition_prefix = conditional_names
                tmp_prefix = []
                for i in condition_prefix:
                    tmp_t = self.global_tokenizer.tokenize(" " + i[0])
                    for j in range(len(tmp_t)):
                        if j == 0:
                            tmp_prefix.append((tmp_t[j], i[1]))
                        else:
                            tmp_prefix.append((tmp_t[j], -1))

                tmp_token_list_with_cluster_ids = tmp_prefix + tmp_token_list_with_cluster_ids

            else:
                tmp_token_list_with_cluster_ids = [['#', -1]] + tmp_token_list_with_cluster_ids

            """ Build the src list ang tgt list for DGL GNN implementation """
            src_list = []
            tgt_list = []
            text_input_list = []
            for k, v in enumerate(tmp_token_list_with_cluster_ids):
                text_input_list.append(v[0])
                if v[1] != -1:
                    find_precedent = [j for j in range(k) if tmp_token_list_with_cluster_ids[j][1] == v[1]]
                    if len(find_precedent) > 0:
                        src_list.append(k)
                        tgt_list.append(max(find_precedent))

            assert len(src_list) == len(tgt_list)

            tmp_line_idx += 1

            """ Truncating the lengthy samples """
            if len(text_input_list) > 1023:
                print("Truncate the lengthy sample:", len(text_input_list))
                cut_num = len([i for i in src_list if i > 1022])
                print(cut_num)
                src_list = src_list[:-cut_num]
                tgt_list = tgt_list[:-cut_num]
                print(src_list)
                print(tgt_list)
                assert len(src_list) == len(tgt_list)

            """ We write all information as a text file """
            text_input_list = text_input_list[:1023]
            output_fp.write(" ".join(text_input_list) + " ##### " + str(self.global_tokenizer.convert_tokens_to_ids(text_input_list)) + \
                            " ##### " + str(src_list) + " ##### " + str(tgt_list) + " ##### " + str(len(text_input_list)) + "\n")

            assert len(text_input_list) == len(self.global_tokenizer.convert_tokens_to_ids(text_input_list))
            assert len(self.global_tokenizer.convert_tokens_to_ids(text_input_list)) < 1024

            if debug:
                for k, v in enumerate(text_input_list):
                    if k in src_list:
                        print(">>>>>>>> ", v.replace("Ġ", ""), k, tgt_list[src_list.index(k)])
                    else:
                        print(v.replace("Ġ", ""), k, "X")
                tmp_matrix = Prev_Coreference_Matrix(len(text_input_list), src_list, tgt_list)
                print(tmp_matrix)

        output_fp.close()
        print("The file is saved in:", task_name + ".source")

