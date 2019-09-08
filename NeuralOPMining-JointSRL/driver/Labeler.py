import torch.nn.functional as F
from driver.Utils import *
from driver.SRLModel import *
import torch.optim.lr_scheduler
from driver.Layer import *
import numpy as np


class SRLLabeler(object):
    def __init__(self, model, srl_model):
        self.model = model
        self.srl_model = srl_model
        p = next(filter(lambda p: p.requires_grad, model.parameters()))
        self.use_cuda = p.is_cuda
        self.device = p.get_device() if self.use_cuda else None

    def parse_one_batch(self, words, extwords, predicts, inmasks, bTrain):
        if bTrain and self.model.config.parser_tune == 1:
            self.srl_model.train()
        else:
            self.srl_model.eval()

        srl_out = self.srl_model.forward(words, extwords, predicts, inmasks)

        return srl_out

    def forward(self, words, extwords, predicts, inmasks, \
                srl_words, srl_extwords, srl_predicts, srl_inmasks):
        if self.use_cuda:
            words, extwords = words.cuda(self.device), extwords.cuda(self.device)
            predicts = predicts.cuda(self.device)
            inmasks = inmasks.cuda(self.device)
            srl_words, srl_extwords = srl_words.cuda(self.device), srl_extwords.cuda(self.device)
            srl_predicts = srl_predicts.cuda(self.device)
            srl_inmasks = srl_inmasks.cuda(self.device)

        srl_outputs = self.parse_one_batch(srl_words, srl_extwords, \
                                           srl_predicts, srl_inmasks, self.model.training)

        label_scores = self.model.forward(words, extwords, predicts, inmasks, srl_outputs)
        # cache
        self.label_scores = label_scores

    def compute_loss(self, answers, outmasks):
        if self.use_cuda:
            answers = answers.cuda(self.device)
            outmasks = outmasks.cuda(self.device)
        loss = self.model.compute_loss(self.label_scores, answers, outmasks)
        stats = self.stats(loss.data[0], self.label_scores.data, answers.data)
        return loss, stats

    def stats(self, loss, scores, target):
        """
        Compute and return a Statistics object.
        Args:
            loss(Tensor): the loss computed by the loss criterion.
            scores(Tensor): a sequence of predict output with scores.
        """
        pred = scores.max(2)[1]
        non_padding = target.ne(self.model.PAD)
        num_words = non_padding.sum()
        num_correct = pred.eq(target).masked_select(non_padding).sum()
        return Statistics(loss, num_words, num_correct)


    def label(self, words, extwords, predicts, inmasks,
              srl_words, srl_extwords, srl_predicts, srl_inmasks):
        if words is not None:
            self.forward(words, extwords, predicts, inmasks, \
                        srl_words, srl_extwords, srl_predicts, srl_inmasks)

        predict_labels = self.model.decode(self.label_scores, inmasks)

        return predict_labels

