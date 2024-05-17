JV,

MALLADI TEJASVI, CS23M036.

CS6910, Fundamentals of Deep Learning Assignment3.

This would be the code base consisting of my Implementation of Encoder-Decoder architecture (with and without attention) and Transformers for Transliterating English words to Telugu.

Dependencies:

    Python : 3.8.18
    Pandas: 2.0.3
    Numpy: 1.22.3
    pytorch: 2.2.2
    matplotlib: 3.7.2

** Please note that the directory with the codes must also have the directory structure aksharantar_sampled/tel/
** The codes have been written with focus on Telugu, however, the work pretty fine with any other language. To change the language, go to the method setup_and_start_expt(...) in train_attention.py or train_vanilla.py and set target_lang to the appropriate language code, for example "hin" for Hindi, etc.

Codes In the Repo:

    JV CS23M036 DL PA3.ipynb
    JV CS23M036 DL PA3_With_Attention.ipynb

    The above two notebooks were used for building the final code, these are not the final code by themselves. Yet these serve two purposes, they uploaded on kaggle to run wandB sweeps and also they carry the history of the code's evolution.

    These two have been unified by creating a single decoder class that automatically uses attention by considering the parameter : use_attention. The corresponding notebook is:

    JV CS23M036 DL PA3_Unified.ipynb
    This again is not the final submission file.

    Now this led into the creation of the following modules:

    Core_Utils.py : The module containing LanguageProcessor and WordDataset classes, which are core in generating batches of data to feed while training/testing.

    Encoder_Decoder_Architecture.py : This module contains the Encoder, Decoder, BahdanauAttention classes which together define and build the Encoder-Decoder architecture with/without attention.

    Machine_Translator.py : This module puts everything together importing from Core_Utils.py and Encoder_Decoder_Architecture.py. This module creates the encoder and decoder, has the methods to train and to compute accuracies/losses for the encoder-decoder model.

    Finally, the train_vanilla.py, corresponds to the Encoder-Decoder model without attention and hence called vanilla. The module imports all of the above modules and just has the code to configure the experiment's hyerparameters and start an experiment. For the sake of the evaluator's convinience I have added code to compute and display the testing accuracy and loss. Towards the end of the module is a chunk of commented code, that could be used to run wandB sweeps, this code directly came from the first two notebooks mentioned above.

    The module train_attention.py is the Encoder-Decoder with attention counterpart of train_vanilla.py

    Please note that in both train_vanilla.py and train_attention.py the default hyperparameters are set to the ones that gave the best validation accuracies as specified in the report.

    The predictions of the vanilla based model are in the file Predictions_Vanilla.csv

    The predictions of the attention based model are in the file predictions_attention.csv

    Execution:

        python train_vanilla.py <commandline args>
        python train_attention.py <commandline args>


    However, these hyperparamters could be changed as required using the following command line args.

    The command line args for train_vanilla.py


    Name	              Default Value	Description

    
        -b,     --batch_size	        64	          Batch size used for training.

        -bid,     --bidirectional	        True	          If True, input would be seen in both directions.

        -dpt,     --dropout	        0.4	          Dropout Probability.

        -es,     --embedding_size	        256	          Dropout Probability.

        -e,     --epochs	        15	          Number of epochs to train neural network.

        -hs,     --hidden_size	        512	          Hidden State dimension of recurrence unit.

        -lr,    --learning_rate	        3e-4              Learning rate used to optimize model parameters

        -nl,   --num_layers	        4	          Number of Recurrent Cell layers.


        -o,     --optimizer	        nadam	          choices: ["rmsprop", "adam", "nadam"]


        -rt,   --rnn_type	        lstm	          The type of recurrent unit.
        
        -tf,   --teacher_forcing	        0.4	          The Teacher forcing probability.

        -w_d,   --weight_decay	        1e-5	          Weight decay used by optimizers.

        -d      --device                None              The device on which the training happens. [When None, the code automatically detects and uses cuda:0 gpu, if available]


    The command line args for train_attention.py are exactly the same as the above, except that the default values would vary.

    Yet, its better to have it listed:

    The command line args for train_attention.py


    Name	              Default Value	Description

    
        -b,     --batch_size	        512	          Batch size used for training.

        -bid,     --bidirectional	        True	          If True, input would be seen in both directions.

        -dpt,     --dropout	        0	          Dropout Probability.

        -es,     --embedding_size	        256	          Dropout Probability.

        -e,     --epochs	        15	          Number of epochs to train neural network.

        -hs,     --hidden_size	        512	          Hidden State dimension of recurrence unit.

        -lr,    --learning_rate	        3e-4              Learning rate used to optimize model parameters

        -nl,   --num_layers	        2	          Number of Recurrent Cell layers.


        -o,     --optimizer	        rmsprop	          choices: ["rmsprop", "adam", "nadam"]


        -rt,   --rnn_type	        gru	          The type of recurrent unit.
        
        -tf,   --teacher_forcing	        0.4	          The Teacher forcing probability.

        -w_d,   --weight_decay	        1e-5	          Weight decay used by optimizers.

        -d      --device                None              The device on which the training happens. [When None, the code automatically detects and uses cuda:0 gpu, if available]

    The Notebook studying_attention.ipynb, was used to study working of attention :

        1. To find words that attention based model got right, but the vanilla model didn't
        2. To plot attention heatmaps.

Main Code (Submission) Files:

    Core_Utils.py
    Encoder_Decoder_Architecture.py
    Machine_Translator.py
    train_vanilla.py
    train_attention.py

Design choices:

The <start> and <end> tokens are defined and indexed in the vocab, but they are not added while actually converting the word pairs into indices. Rather a <pad> token is added to match the length of the longest string in the batch.

This works, because strings always end with a <pad>, making it as good as an <end> token. And <start> is passed as the input at the first step of decoder.

Another Design choice is that, I am tying the values of the embedding dimension to that of hidden dimension, however, the code still supports using an independent value for embedding dim. by setting embedding size to the method parameter we can have embedding to be independent.

Thanks.