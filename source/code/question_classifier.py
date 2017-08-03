###############################################################################
# Author: Wasi Ahmad
# Project: Quora Duplicate Question Detection
# Date Created: 7/25/2017
#
# File Description: This script contains code related to quora duplicate
# question classifier.
###############################################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_layer import EmbeddingLayer, Encoder


class QuoraRNN(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embeddings_index, args, select_method='max'):
        """"Constructor of the class."""
        super(QuoraRNN, self).__init__()
        self.config = args
        self.feature_select_method = select_method
        self.num_directions = 2 if args.bidirection else 1

        self.embedding = EmbeddingLayer(len(dictionary), self.config)
        self.embedding.init_embedding_weights(dictionary, embeddings_index, self.config.emsize)

        self.encoder = Encoder(self.config.emsize, self.config.nhid, self.config.bidirection, self.config)
        self.dropout = nn.Dropout(self.config.dropout)
        self.dense1 = nn.Linear(self.config.nhid * self.num_directions * 2, self.config.nhid * self.num_directions)
        self.dense2 = nn.Linear(self.config.nhid * self.num_directions, 2)

    def forward(self, batch_sentence1, batch_sentence2):
        """"Defines the forward computation of the question classifier."""
        embedded1 = self.dropout(self.embedding(batch_sentence1))
        embedded2 = self.dropout(self.embedding(batch_sentence2))

        if self.config.model == 'LSTM':
            # For the first sentences in batch
            encoder_hidden1, encoder_cell1 = self.encoder.init_weights(batch_sentence1.size(0))
            output1, hidden1 = self.encoder(embedded1, (encoder_hidden1, encoder_cell1))
            # For the second sentences in batch
            encoder_hidden2, encoder_cell2 = self.encoder.init_weights(batch_sentence2.size(0))
            output2, hidden2 = self.encoder(embedded2, (encoder_hidden2, encoder_cell2))
        else:
            # For the first sentences in batch
            encoder_hidden1 = self.encoder.init_weights(batch_sentence1.size(0))
            output1, hidden1 = self.encoder(embedded1, encoder_hidden1)
            # For the second sentences in batch
            encoder_hidden2 = self.encoder.init_weights(batch_sentence2.size(0))
            output2, hidden2 = self.encoder(embedded2, encoder_hidden2)

        assert output1.size() == output2.size()

        if self.feature_select_method == 'max':
            encoded_questions1 = torch.max(output1, 1)[0].squeeze()
            encoded_questions2 = torch.max(output2, 1)[0].squeeze()
        elif self.feature_select_method == 'average':
            encoded_questions1 = torch.sum(output1, 1).squeeze() / batch_sentence1.size(1)
            encoded_questions2 = torch.sum(output2, 1).squeeze() / batch_sentence2.size(1)
        elif self.feature_select_method == 'last':
            encoded_questions1 = output1[:, -1, :]
            encoded_questions2 = output2[:, -1, :]

        # compute angle between question representation
        angle = torch.mul(encoded_questions1, encoded_questions2)
        # compute distance between question representation
        distance = torch.abs(encoded_questions1 - encoded_questions2)
        # combined_representation = batch_size x (hidden_size * num_directions * 2)
        combined_representation = torch.cat((angle, distance), 1)

        return F.log_softmax(self.dense2(F.relu(self.dense1(combined_representation))))


# taken from https://github.com/facebookresearch/InferSent/blob/master/models.py#L637
class ConvNetEncoder(nn.Module):
    """Class that classifies question pair as duplicate or not."""

    def __init__(self, dictionary, embeddings_index, config):
        super(ConvNetEncoder, self).__init__()
        self.config = config

        self.embedding = EmbeddingLayer(len(dictionary), self.config)
        self.embedding.init_embedding_weights(dictionary, embeddings_index, self.config.emsize)
        self.dropout = nn.Dropout(self.config.dropout)

        self.convnet1 = nn.Sequential(
            nn.Conv1d(self.config.emsize, self.config.nhid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convnet2 = nn.Sequential(
            nn.Conv1d(self.config.nhid, self.config.nhid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convnet3 = nn.Sequential(
            nn.Conv1d(self.config.nhid, self.config.nhid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.convnet4 = nn.Sequential(
            nn.Conv1d(self.config.nhid, self.config.nhid, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        self.dense1 = nn.Linear(self.config.nhid * 4 * 2, self.config.nhid * 2)
        self.dense2 = nn.Linear(self.config.nhid * 2, 2)

    def forward(self, batch_sentence1, batch_sentence2):
        embedded1 = self.dropout(self.embedding(batch_sentence1))
        embedded2 = self.dropout(self.embedding(batch_sentence2))

        questions1 = embedded1.transpose(1, 2).contiguous()
        questions1 = self.convnet1(questions1)
        u1 = torch.max(questions1, 2)[0].squeeze(2)
        questions1 = self.convnet2(questions1)
        u2 = torch.max(questions1, 2)[0].squeeze(2)
        questions1 = self.convnet3(questions1)
        u3 = torch.max(questions1, 2)[0].squeeze(2)
        questions1 = self.convnet4(questions1)
        u4 = torch.max(questions1, 2)[0].squeeze(2)
        question1_rep = torch.cat((u1, u2, u3, u4), 1)

        questions2 = embedded2.transpose(1, 2).contiguous()
        questions2 = self.convnet1(questions2)
        u1 = torch.max(questions2, 2)[0].squeeze(2)
        questions2 = self.convnet2(questions2)
        u2 = torch.max(questions2, 2)[0].squeeze(2)
        questions2 = self.convnet3(questions2)
        u3 = torch.max(questions2, 2)[0].squeeze(2)
        questions2 = self.convnet4(questions2)
        u4 = torch.max(questions2, 2)[0].squeeze(2)
        question2_rep = torch.cat((u1, u2, u3, u4), 1)

        # compute angle between question representation
        angle = torch.mul(question1_rep, question2_rep)
        # compute distance between question representation
        distance = torch.abs(question1_rep - question2_rep)
        # combined_representation = batch_size x (hidden_size * num_directions * 2)
        combined_representation = torch.cat((angle, distance), 1)

        return F.log_softmax(self.dense2(F.relu(self.dense1(combined_representation))))
