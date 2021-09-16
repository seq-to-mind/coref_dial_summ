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
3. One can obtain the corefernece resolution from the model with script:  
   `./dialogue_coreference/dialogue_coreference.py`
5. For dialogue coreference resolution post-processing, one can call the function in the file:  
   `./dialogue_coreference/reading_and_writing_as_input_keep_SPAN.py`
7. For end-to-end conversation samples construction with coreference information, please run or refer to the script:  
   `./dialogue_coreference/end2end_build_data.py`
9. Noted that the processed samples will be tokenized via the RoBERTa/BART sub-word tokenization.

## Outputs for SAMSum Corpus
See the model predictions of SAMSum test set in `./model_outputs/`

## Citation

```
@inproceedings{liu-etal-2021-coreference,
    title = "Coreference-Aware Dialogue Summarization",
    author = "Liu, Zhengyuan  and
      Shi, Ke  and
      Chen, Nancy",
    booktitle = "Proceedings of the 22nd Annual Meeting of the Special Interest Group on Discourse and Dialogue",
    month = jul,
    year = "2021",
    address = "Singapore and Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.sigdial-1.53",
    pages = "509--519",
}
```

