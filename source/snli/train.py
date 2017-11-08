###############################################################################
# Author: Wasi Ahmad
# Project: Quora Duplicate Question Detection
# Date Created: 7/25/2017
#
# File Description: This script contains code to train the model.
###############################################################################

import time, helper, torch
import torch.nn as nn


class Train:
    """Train class that encapsulate all functionalities of the training procedure."""

    def __init__(self, model, optimizer, dictionary, embeddings_index, config, best_loss):
        self.model = model
        self.dictionary = dictionary
        self.embeddings_index = embeddings_index
        self.config = config
        self.criterion = nn.CrossEntropyLoss()

        self.optimizer = optimizer
        self.best_dev_loss = best_loss
        self.times_no_improvement = 0
        self.stop = False
        self.train_losses = []
        self.dev_losses = []

    def train_epochs(self, train_corpus, dev_corpus, start_epoch, n_epochs):
        """Trains model for n_epochs epochs"""
        for epoch in range(start_epoch, start_epoch + n_epochs):
            if not self.stop:
                print('\nTRAINING : Epoch ' + str((epoch + 1)))
                self.train(train_corpus)
                # training epoch completes, now do validation
                print('\nVALIDATING : Epoch ' + str((epoch + 1)))
                # dev_loss = self.validate(dev_corpus)
                dev_acc = self.validate_in_acc(dev_corpus)
                self.dev_losses.append(dev_acc)
                print('validation acc = %.4f' % dev_acc)
                # save model if dev loss goes down
                if self.best_dev_loss == -1 or self.best_dev_loss < dev_acc:
                    self.best_dev_loss = dev_acc
                    helper.save_checkpoint({
                        'epoch': (epoch + 1),
                        'state_dict': self.model.state_dict(),
                        'best_loss': self.best_dev_loss,
                        'optimizer': self.optimizer.state_dict(),
                    }, self.config.save_path + 'model_best.pth.tar')
                    torch.save(self.model, self.config.save_path + 'model.pickle')
                    self.times_no_improvement = 0
                else:
                    self.times_no_improvement += 1
                    # no improvement in validation loss for last n iterations, so stop training
                    if self.times_no_improvement == 3:
                        self.stop = True
                # save the train and development loss plot
                helper.save_plot(self.train_losses, self.config.save_path, 'training', epoch + 1)
                helper.save_plot(self.dev_losses, self.config.save_path, 'dev', epoch + 1)
            else:
                break

    def train(self, train_corpus):
        # Turn on training mode which enables dropout.
        self.model.train()

        # Splitting the data in batches
        train_batches = helper.batchify(train_corpus.data, self.config.batch_size)
        print('number of train batches = ', len(train_batches))

        start = time.time()
        print_loss_total = 0
        plot_loss_total = 0

        num_batches = len(train_batches)
        for batch_no in range(1, num_batches + 1):
            # Clearing out all previous gradient computations.
            self.optimizer.zero_grad()
            train_sentences1, sent_len1, train_sentences2, sent_len2, train_labels = helper.batch_to_tensors(train_batches[batch_no - 1],
                                                                                       self.dictionary)
            if self.config.cuda:
                train_sentences1 = train_sentences1.cuda()
                train_sentences2 = train_sentences2.cuda()
                train_labels = train_labels.cuda()

            assert train_sentences1.size(0) == train_sentences2.size(0)

            softmax_out = self.model(train_sentences1, sent_len1, train_sentences2, sent_len2)
            loss = self.criterion(softmax_out, train_labels)
            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = loss.mean()
            loss.backward()
            self.optimizer.step()

            print_loss_total += loss.data[0]
            plot_loss_total += loss.data[0]

            if batch_no % self.config.print_every == 0:
                print_loss_avg = print_loss_total / self.config.print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (
                    helper.show_progress(start, batch_no / num_batches), batch_no,
                    batch_no / num_batches * 100, print_loss_avg))

            if batch_no % self.config.plot_every == 0:
                plot_loss_avg = plot_loss_total / self.config.plot_every
                self.train_losses.append(plot_loss_avg)
                plot_loss_total = 0

    def validate(self, dev_corpus):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        dev_batches = helper.batchify(dev_corpus.data, self.config.batch_size)
        print('number of dev batches = ', len(dev_batches))

        num_batches = len(dev_batches)
        avg_loss = 0
        for batch_no in range(1, num_batches + 1):
            dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_tensors(dev_batches[batch_no - 1],
                                                                                 self.dictionary)
            if self.config.cuda:
                dev_sentences1 = dev_sentences1.cuda()
                dev_sentences2 = dev_sentences2.cuda()
                dev_labels = dev_labels.cuda()

            assert dev_sentences1.size(0) == dev_sentences2.size(0)

            softmax_out = self.model(dev_sentences1, sent_len1, dev_sentences2, sent_len2)
            loss = self.criterion(softmax_out, dev_labels)
            # Important if we are using nn.DataParallel()
            if loss.size(0) > 1:
                loss = loss.mean()
            avg_loss += loss.data[0]

        return avg_loss / num_batches

    def validate_in_acc(self, dev_corpus):
        # Turn on evaluation mode which disables dropout.
        self.model.eval()

        dev_batches = helper.batchify(dev_corpus.data, self.config.batch_size)
        print('number of dev batches = ', len(dev_batches))

        num_batches = len(dev_batches)
        n_correct, n_total = 0, 0
        for batch_no in range(1, num_batches + 1):
            dev_sentences1, sent_len1, dev_sentences2, sent_len2, dev_labels = helper.batch_to_tensors(dev_batches[batch_no - 1],
                                                                                 self.dictionary)
            if self.config.cuda:
                dev_sentences1 = dev_sentences1.cuda()
                dev_sentences2 = dev_sentences2.cuda()
                dev_labels = dev_labels.cuda()

            assert dev_sentences1.size(0) == dev_sentences2.size(0)

            softmax_prob = self.model(dev_sentences1, sent_len1, dev_sentences2, sent_len2)
            n_correct += (torch.max(softmax_prob, 1)[1].view(dev_labels.size()).data == dev_labels.data).sum()
            n_total += len(dev_batches[batch_no - 1])

        return 100. * n_correct / n_total
