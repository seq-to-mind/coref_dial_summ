1. We use the # as the utterance segmentation token.
2. For multiple coreference resolution outputs, the utterance segmentation token with be replaced by newline, period and semi-colon.
3. We provided the samples after data pre-processing, and the post-processing of the coreference resolution.
4. The processed data description:  
  For instance, each row in the file `test.source` contains information as below:  
  Text Tokens after BART tokenization ##### Token IDs after BART tokenization ##### Start Token ID of One Coreference-Linked Span ##### Target Token ID of One Coreference-Linked Span ##### Token Number after BART tokenization
