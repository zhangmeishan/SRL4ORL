# !/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import math
import sys

def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


class Statistics(object):
    """
    Train/validate loss statistics.
    """
    def __init__(self, loss=0, n_words=0, n_correct=0):
        self.loss = loss
        self.n_words = n_words
        self.n_correct = n_correct
        self.n_src_words = 0
        self.start_time = time.time()

    def update(self, stat):
        self.loss += stat.loss * stat.n_words
        self.n_words += stat.n_words
        self.n_correct += stat.n_correct

    def ppl(self):
        return safe_exp(self.loss / self.n_words)

    def accuracy(self):
        return 100 * (self.n_correct / self.n_words)

    def elapsed_time(self):
        return time.time() - self.start_time

    def print_out(self, step, epoch, batch, n_batches):
        t = self.elapsed_time()
        out_info = ("Step %d, Epoch %d, %d/%d| acc: %.2f| ppl: %.2f| %.1f tgt tok/s| %.2f s elapsed") \
                   % (step, epoch, batch, n_batches,self.accuracy(), self.ppl(), \
                    self.n_words / (t + 1e-5), time.time() - self.start_time)
        print(out_info)
        sys.stdout.flush()

    def print_valid(self, step):
        t = self.elapsed_time()
        out_info = ("Valid at step %d: acc %.2f, ppl: %.2f, %.1f tgt tok/s, %.2f s elapsed") % \
              (step, self.accuracy(), self.ppl(), self.n_words / (t + 1e-5),
               time.time() - self.start_time)
        print(out_info)
        sys.stdout.flush()
