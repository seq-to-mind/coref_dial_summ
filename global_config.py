

use_cuda = True
gpu_id = "0"

pretrained_model = "facebook/bart-large"
pretrained_tokenizer = "facebook/bart-large"

corpus_mode = "SAMSum"

data_path = "data/SAMsum_data/"
pre_tokenized_samples = True

freeze_some_bert_layer = False
different_learning_rate = False
scheduling_learning_rate = False

using_label_smoothing = True
smooth_epsilon = 0.1

start_from_epoch = 0
train = True
save_model = False
load_model_path = "saved_models/XXX"

batch_loss_print_interval = 20
print_all_predictions = True

hidden_size = 768

min_gen_len = 8
max_gen_len = 100

batch_size = 2
num_epochs = 7
learning_rate = 0.00002

add_coref_attn_layer = True
replace_coref_head = False
coref_head_probe = False
