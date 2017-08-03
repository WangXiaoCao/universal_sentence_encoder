###############################################################################
# Author: Wasi Ahmad
# Project: Quora Duplicate Question Detection
# Date Created: 7/25/2017
#
# File Description: This script contains code to test the model.
###############################################################################

import util, helper, data, torch
from question_classifier import QuoraRNN, ConvNetEncoder

args = util.get_args()


def evaluate(model, batches, dictionary):
    """Evaluate question classifier model on test data."""
    # Turn on evaluation mode which disables dropout.
    model.eval()

    n_correct, n_total = 0, 0
    for batch_no in range(len(batches)):
        test_sentences1, test_sentences2, test_labels = helper.batch_to_tensors(batches[batch_no], dictionary)
        if args.cuda:
            test_sentences1 = test_sentences1.cuda()
            test_sentences2 = test_sentences2.cuda()
            test_labels = test_labels.cuda()
        assert test_sentences1.size(0) == test_sentences1.size(0)

        softmax_prob = model(test_sentences1, test_sentences2)
        n_correct += (torch.max(softmax_prob, 1)[1].view(test_labels.size()).data == test_labels.data).sum()
        n_total += len(batches[batch_no])

    return 100. * n_correct / n_total


if __name__ == "__main__":
    dictionary = helper.load_object(args.data + 'dictionary.p')
    test_corpus = data.Corpus(args.data + 'quora/', 'quora_duplicate_questions_test.tsv', dictionary, True)
    print('Test set size = ', len(test_corpus.data))

    embeddings_index = helper.load_word_embeddings(args.word_vectors_directory, 'glove.840B.300d.quora.dup.txt')
    model = ConvNetEncoder(dictionary, embeddings_index, args)
    # model = QuoraRNN(dictionary, embeddings_index, args, select_method='last')
    if args.cuda:
        model = model.cuda()
    helper.load_model_states_from_checkpoint(model, args.pretrained_path + 'model_best.pth.tar', 'state_dict')
    print('Vocabulary size = ', len(dictionary))

    test_batches = helper.batchify(test_corpus.data, args.batch_size)
    test_accuracy = evaluate(model, test_batches, dictionary)
    print('Test Accuracy: %f%%' % test_accuracy)
