## Overview

### Requirement

* python 3.5
* pytorch (0.1.6)
* numpy 1.11.3
* [GloVe 300d word embeddings (840B)](https://nlp.stanford.edu/projects/glove/)

### Command Line Arguments

The main.py script accepts the following arguments:

```
optional arguments:
  -h, --help            show this help message and exit
  --data DATA           location of the data corpus, default = '../data/'
  --model MODEL         type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
  --bidirection         use bidirectional recurrent unit
  --emsize EMSIZE       size of word embeddings
  --nhid NHID           humber of hidden units per layer
  --nlayers NLAYERS     number of layers
  --lr LR               initial learning rate
  --lr_decay            decay ratio for learning rate
  --clip CLIP           gradient clipping
  --epochs EPOCHS       upper epoch limit
  --batch_size N        batch size
  --dropout DROPOUT     dropout applied to layers (0 = no dropout)
  --seed SEED           random seed
  --cuda                use CUDA
  --print_every N       training report interval
  --plot_every          plotting interval
  --save_path           path to save the final model
  ```

