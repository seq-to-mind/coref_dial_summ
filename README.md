## Introduction
One implementation of the paper "Coreference-Aware Dialogue Summarization".

## Package Requirements
1. pytorch==1.7.1
2. transformers==4.8.2
3. click==7.1.2
4. sentencepiece==0.1.92
5. allennlp==2.6.0
6. allennlp-models==2.6.0

## Dialogue Coreference Resolution
1. Download the off-the-shelf model from AllenNLP:  
   `allennlp-public-models/coref-spanbert-large-2021.03.10`
3. You can obtain the coreference resolution from the model with the script:  
   `./dialogue_coreference/dialogue_coreference.py`
5. For dialogue coreference resolution post-processing, you can call the function in the file:  
   `./dialogue_coreference/reading_and_writing_as_input_keep_SPAN.py`
7. For end-to-end conversation samples construction with coreference information, please run or refer to the script:  
   `./dialogue_coreference/end2end_build_data.py`
9. Noted that the processed samples will be tokenized via the RoBERTa/BART sub-word tokenization.

## Coref-Aware Summarization
1. The data after dialogue coreference resolution can be used to train the coref-aware summarizer.
2. You can read the samples in text format, then read the tokenized id/coreference information from each row.  
  For instance, each row in the file `train.source` contains information as below:  
  Text Tokens after BART tokenization ##### Token IDs after BART tokenization ##### Start Token ID of One Coreference-Linked Span ##### Target Token ID of One Coreference-Linked Span ##### Token Number after BART tokenization
3. For our implementation, you will need to replace the original `generation_utils.py` and `modeling_bart.py` in the `Transformers` library, with the corresponding files in this repo.
4. You can search the keyword 'coref' in our updated `generation_utils.py` and `modeling_bart.py` to see the implementation details.
5. See running configurations in the `global_config.py` file.
6. We provide the `self-attention-layer` and `Transformer head manipulation` methods to incorporate coreference information, which are computationally efficient.


## Generated Summaries for SAMSum Corpus
See the predictions from the coreference-aware summarization model of SAMSum test set in `./model_outputs/`

## Citation

```
@inproceedings{liu-etal-2021-coreference,
    title = "Coreference-Aware Dialogue Summarization",
    author = "Liu, Zhengyuan and Shi, Ke and Chen, Nancy",
    booktitle = "Proceedings of the 22nd Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2021",
    address = "Singapore and Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.sigdial-1.53",
    pages = "509--519",
}
```

