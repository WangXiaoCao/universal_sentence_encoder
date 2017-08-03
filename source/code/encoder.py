###############################################################################
# Author: Wasi Ahmad
# Project: Quora Duplicate Question Detection
# Date Created: 7/25/2017
#
# File Description: This script contains code related to the Encoder class.
###############################################################################

import torch.nn as nn
from torch.autograd import Variable


class Encoder(nn.Module):
    """Encoder class of a sequence-to-sequence network"""

    def __init__(self, input_size, config):
        """"Constructor of the class"""
        super(Encoder, self).__init__()
        self.config = config
        self.drop = nn.Dropout(self.config.dropout)

        if self.config.model in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, self.config.model)(input_size, self.config.nhid, self.config.nlayers,
                                                      batch_first=True, dropout=self.config.dropout,
                                                      bidirectional=self.config.bidirection)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[self.config.model]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                         options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(input_size, self.config.nhid, self.config.nlayers, nonlinearity=nonlinearity,
                              batch_first=True, dropout=self.config.dropout, bidirectional=self.config.bidirection)

    def forward(self, input_variable, hidden):
        """"Defines the forward computation of the encoder"""
        # input to rnn layers requires: seq_len x batch_size x input_size
        output = input_variable
        for i in range(self.config.nlayers):
            output, hidden = self.rnn(output, hidden)
            output = self.drop(output)
        return output, hidden

    def init_weights(self, bsz):
        """Initialize weight parameters for the encoder."""
        weight = next(self.parameters()).data
        num_directions = 2 if self.config.bidirection else 1
        if self.config.model == 'LSTM':
            return Variable(weight.new(self.config.nlayers * num_directions, bsz, self.config.nhid).zero_()), Variable(
                weight.new(self.config.nlayers * num_directions, bsz, self.config.nhid).zero_())
        else:
            return Variable(weight.new(self.config.nlayers * num_directions, bsz, self.config.nhid).zero_())
