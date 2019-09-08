from data.Vocab import *
import numpy as np
import torch
from torch.autograd import Variable
import argparse

def read_corpus(file_path):
    data = []
    with open(file_path, 'r') as infile:
        for sentence in readSRL(infile):
            data.append(sentence)
    return data

def sentences_numberize(sentences, vocab, srl_vocab):
    for sentence in sentences:
        yield sentence2id(sentence, vocab, srl_vocab)

def sentence2id(sentence, vocab, srl_vocab):
    tokens = []
    index = 0
    for token in sentence.words:
        wordid = vocab.word2id(token.form)
        extwordid = vocab.extword2id(token.form)
        srl_wordid = srl_vocab.word2id(token.form)
        srl_extwordid = srl_vocab.extword2id(token.form)
        if index < sentence.key_start or index > sentence.key_end:
            labelid = vocab.label2id(token.label)
        else:
            labelid = vocab.PAD
        tokens.append([wordid, extwordid, labelid, srl_wordid, srl_extwordid])
        index = index + 1

    return tokens,sentence.key_head,sentence.key_start,sentence.key_end

def batch_slice(data, batch_size, bsorted=True):
    batch_num = int(np.ceil(len(data) / float(batch_size)))
    for i in range(batch_num):
        cur_batch_size = batch_size if i < batch_num - 1 else len(data) - batch_size * i
        sentences = [data[i * batch_size + b] for b in range(cur_batch_size)]

        batch_size = len(sentences)
        src_ids = list(range(batch_size))
        if bsorted:
            src_ids = sorted(range(batch_size), key=lambda src_id: sentences[src_id].length, reverse=True)

        sorted_sentences = [sentences[src_id] for src_id in src_ids]

        yield sorted_sentences


def data_iter(data, batch_size, shuffle=True, bsorted=True):
    """
    randomly permute data, then sort by source length, and partition into batches
    ensure that the length of  sentences in each batch
    """

    batched_data = []
    if shuffle: np.random.shuffle(data)
    batched_data.extend(list(batch_slice(data, batch_size, bsorted)))

    if shuffle: np.random.shuffle(batched_data)
    for batch in batched_data:
        yield batch


def batch_data_variable(batch, vocab, srl_vocab):
    length = batch[0].length
    batch_size = len(batch)
    for b in range(1, batch_size):
        if batch[b].length > length: length = batch[b].length

    words = Variable(torch.LongTensor(batch_size, length).fill_(vocab.PAD), requires_grad=False)
    extwords = Variable(torch.LongTensor(batch_size, length).fill_(vocab.PAD), requires_grad=False)
    predicts = Variable(torch.LongTensor(batch_size, length).fill_(vocab.PAD), requires_grad=False)
    inmasks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)
    labels = Variable(torch.LongTensor(batch_size, length).fill_(vocab.PAD), requires_grad=False)
    outmasks = Variable(torch.LongTensor(batch_size, length).zero_(), requires_grad=False)

    srl_words = Variable(torch.LongTensor(batch_size, length).fill_(srl_vocab.PAD), requires_grad=False)
    srl_extwords = Variable(torch.LongTensor(batch_size, length).fill_(srl_vocab.PAD), requires_grad=False)
    srl_predicts = Variable(torch.LongTensor(batch_size, length).fill_(srl_vocab.PAD), requires_grad=False)
    srl_inmasks = Variable(torch.Tensor(batch_size, length).zero_(), requires_grad=False)

    b = 0
    for tokens, key_head, key_start, key_end in sentences_numberize(batch, vocab, srl_vocab):
        index = 0
        for word in tokens:
            words[b, index] = word[0]
            extwords[b, index] = word[1]
            labels[b, index] = word[2]
            inmasks[b, index] = 1
            outmasks[b, index] = 1
            predicts[b, index] = 2
            if index >= key_start and index <= key_end:
                predicts[b, index] = 1

            srl_words[b, index] = word[3]
            srl_extwords[b, index] = word[4]
            srl_inmasks[b, index] = 1
            srl_predicts[b, index] = 2

            #if index >= key_start and index <= key_end:
            if index == key_head:
                srl_predicts[b, index] = 1

            index += 1
        b += 1

    return words, extwords, predicts, inmasks, labels, outmasks.byte(), \
           srl_words, srl_extwords, srl_predicts, srl_inmasks

def batch_variable_srl(inputs, labels, vocab):
    for input, label in zip(inputs, labels):
        predicted_labels = []
        for idx in range(input.length):
            if idx < input.key_start or idx > input.key_end:
                predicted_labels.append(vocab.id2label(label[idx]))
            else:
                predicted_labels.append(input.words[idx].label)
        normed_labels, modifies = normalize_labels(predicted_labels)
        tokens = []
        for idx in range(input.length):
            tokens.append(Word(idx, input.words[idx].org_form, normed_labels[idx]))
        yield Sentence(tokens)

