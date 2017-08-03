###############################################################################
# Author: Wasi Ahmad
# Project: Quora Duplicate Question Detection
# Date Created: 7/25/2017
#
# File Description: This script provides general purpose utility functions that
# are required at different steps in the experiments.
###############################################################################

import re, os, pickle, string, math, time, util, torch, glob
import numpy as np
from nltk import wordpunct_tokenize
from numpy.linalg import norm
from torch.autograd import Variable
import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from collections import OrderedDict

args = util.get_args()


def normalize_word_embedding(v):
    return np.array(v) / norm(np.array(v))


def load_word_embeddings(directory, file):
    embeddings_index = {}
    f = open(os.path.join(directory, file))
    for line in f:
        try:
            values = line.split()
            word = values[0]
            embeddings_index[word] = normalize_word_embedding([float(x) for x in values[1:]])
        except ValueError as e:
            print(e)
    f.close()
    return embeddings_index


def save_word_embeddings(directory, file, embeddings_index, words):
    f = open(os.path.join(directory, file), 'w')
    for word in words:
        if word in embeddings_index:
            f.write(word + '\t' + '\t'.join(str(x) for x in embeddings_index[word]) + '\n')
    f.close()


def save_checkpoint(state, filename='./checkpoint.pth.tar'):
    if os.path.isfile(filename):
        os.remove(filename)
    torch.save(state, filename)


def load_model_states_from_checkpoint(model, filename, tag):
    """Load model states from a previously saved checkpoint."""
    assert os.path.exists(filename)
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint[tag])


def load_model_states_without_dataparallel(model, filename, tag):
    """Load a previously saved model states."""
    assert os.path.exists(filename)
    checkpoint = torch.load(filename)
    new_state_dict = OrderedDict()
    for k, v in checkpoint[tag].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)


def save_object(obj, filename):
    """Save an object into file."""
    with open(filename, 'wb') as output:
        pickle.dump(obj, output)


def load_object(filename):
    """Load object from file."""
    with open(filename, 'rb') as input:
        obj = pickle.load(input)
    return obj


def tokenize_and_normalize(s):
    """Tokenize and normalize string."""
    token_list = []
    tokens = wordpunct_tokenize(s.lower())
    token_list.extend([x for x in tokens if not re.fullmatch('[' + string.punctuation + ']+', x)])
    return token_list


def initialize_out_of_vocab_words(dimension):
    """Returns a random vector of size dimension where mean is 0 and standard deviation is 1."""
    return np.random.normal(size=dimension)


def sentence_to_tensor(sentence, max_sent_length, dictionary):
    sen_rep = torch.LongTensor(max_sent_length).zero_()
    for i in range(len(sentence)):
        word = sentence[i]
        if word in dictionary.word2idx:
            sen_rep[i] = dictionary.word2idx[word]
        else:
            sen_rep[i] = dictionary.word2idx[dictionary.unknown_token]
    return sen_rep


def batch_to_tensors(batch, dictionary):
    """Convert a list of sequences to a list of tensors."""
    max_sent_length = 0
    for item in batch:
        if max_sent_length < len(item.sentence1):
            max_sent_length = len(item.sentence1)
        if max_sent_length < len(item.sentence2):
            max_sent_length = len(item.sentence2)

    all_sentences1 = torch.LongTensor(len(batch), max_sent_length)
    all_sentences2 = torch.LongTensor(len(batch), max_sent_length)
    labels = torch.LongTensor(len(batch))
    for i in range(len(batch)):
        all_sentences1[i] = sentence_to_tensor(batch[i].sentence1, max_sent_length, dictionary)
        all_sentences2[i] = sentence_to_tensor(batch[i].sentence2, max_sent_length, dictionary)
        labels[i] = batch[i].label
    return Variable(all_sentences1), Variable(all_sentences2), Variable(labels)


def batchify(data, bsz):
    """Transform data into batches."""
    np.random.shuffle(data)
    nbatch = len(data) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data[0:nbatch * bsz]
    # Evenly divide the data across the bsz batches.
    batched_data = [[data[bsz * i + j] for j in range(bsz)] for i in range(nbatch)]
    return batched_data


def save_plot(points, filepath, filetag, epoch):
    """Generate and save the plot"""
    path_prefix = os.path.join(filepath, filetag + '_loss_plot_')
    path = path_prefix + 'epoch_{}.png'.format(epoch)
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2)  # this locator puts ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    ax.plot(points)
    fig.savefig(path)
    plt.close(fig)  # close the figure
    for f in glob.glob(path_prefix + '*'):
        if f != path:
            os.remove(f)


def convert_to_minutes(s):
    """Converts seconds to minutes and seconds"""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def show_progress(since, percent):
    """Prints time elapsed and estimated time remaining given the current time and progress in %"""
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (convert_to_minutes(s), convert_to_minutes(rs))
