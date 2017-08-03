###############################################################################
# Author: Wasi Ahmad
# Project: Quora Duplicate Question Detection
# Date Created: 7/25/2017
#
# File Description: This is the main script from where all experimental
# execution begins.
###############################################################################

from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser(description='quora_duplicate_question_detection')
    parser.add_argument('--data', type=str, default='../data/',
                        help='location of the training data')
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_Tanh, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--bidirection', action='store_true',
                        help='use bidirectional recurrent unit')
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=1024,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=1,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=.001,
                        help='initial learning rate')
    parser.add_argument('--lr_decay', type=float, default=.99,
                        help='decay ratio for learning rate')
    parser.add_argument('--clip', type=float, default=5.0,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=100,
                        help='upper limit of epoch')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='batch size')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed for reproducibility')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA for computation')
    parser.add_argument('--print_every', type=int, default=2000, metavar='N',
                        help='training report interval')
    parser.add_argument('--plot_every', type=int, default=2000,
                        help='plotting interval')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='resume from last checkpoint (default: none)')
    parser.add_argument('--resume_snapshot', type=str, default='',
                        help='resume execution from previously saved model')
    parser.add_argument('--save_path', type=str, default='../output/',
                        help='path to save the final model')
    parser.add_argument('--pretrained_path', type=str, default='../trained_models/',
                        help='path where pretrained models are saved')
    parser.add_argument('--word_vectors_file', type=str, default='glove.840B.300d.txt',
                        help='GloVe word embedding version')
    parser.add_argument('--word_vectors_directory', type=str, default='../data/glove/',
                        help='Path of GloVe word embeddings')

    args = parser.parse_args()
    return args
