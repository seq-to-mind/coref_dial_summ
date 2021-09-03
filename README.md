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
1. Download the off-the-shelf model: "allennlp-public-models/coref-spanbert-large-2021.03.10".
2. One can obtain the corefernece resolution from the model with script: 
3. For data post-processing, one can run the script:
4. For end-to-end conversation samples construction with coreference information, please refer to the file:
5. Noted that the processed samples will be tokenized via the RoBERTa/BART sub-word tokenization.

