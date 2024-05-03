JV,

MALLADI TEJASVI, CS23M036.

CS6910, Fundamentals of Deep Learning Assignment3.

This would be the code base consisting of my Implementation of Encoder-Decoder architecture (with and without attention) and Transformers for Transliterating English words to Telugu.


A design choice:

The <start> and <end> tokens are defined and indexed in the vocab, but they are not added while actually converting the word pairs into indices. Rather a <pad> token is added to match the length of the longest string in the batch.

This works, because strings always end with a <pad>, making it as good as an <end> token. And <start> is passed as the input at the first step of decoder.