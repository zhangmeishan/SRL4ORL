import sys
sys.path.extend(["../../","../","./"])
import time
import torch.optim.lr_scheduler
import torch.nn as nn
import random
import argparse
from driver.Config import *
from driver.Model import *
from driver.Labeler import *
from data.Dataloader import *
import pickle
import os
import re
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(data, dev_data, test_data, labeler, vocab, config):
    optimizer = Optimizer(filter(lambda p: p.requires_grad, labeler.model.parameters()), config)

    global_step = 0
    best_score = -1
    batch_num = int(np.ceil(len(data) / float(config.train_batch_size)))
    for iter in range(config.train_iters):
        total_stats = Statistics()
        print('Iteration: ' + str(iter))
        batch_iter = 0

        for onebatch in data_iter(data, config.train_batch_size, True):
            words, extwords, srltags, predicts, inmasks, labels, outmasks = \
                batch_data_variable(onebatch, vocab)
            labeler.model.train()

            labeler.forward(words, extwords, srltags, predicts, inmasks)
            loss, stat = labeler.compute_loss(labels, outmasks)
            loss = loss / config.update_every
            loss.backward()

            total_stats.update(stat)
            total_stats.print_out(global_step, iter, batch_iter, batch_num)
            batch_iter += 1
            if batch_iter % config.update_every == 0 or batch_iter == batch_num:
                nn.utils.clip_grad_norm(filter(lambda p: p.requires_grad, labeler.model.parameters()), \
                                        max_norm=config.clip)
                optimizer.step()
                labeler.model.zero_grad()
                global_step += 1

            if batch_iter % config.validate_every == 0 or batch_iter == batch_num:
                gold_num, predict_num, correct_num = \
                    evaluate(dev_data, labeler, vocab, config.dev_file + '.' + str(global_step))
                dev_score = 200.0 * correct_num / (gold_num + predict_num) if correct_num > 0 else 0.0
                print("Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (correct_num, gold_num, 100.0 * correct_num / gold_num if correct_num > 0 else 0.0,  \
                       correct_num, predict_num, 100.0 * correct_num / predict_num if correct_num > 0 else 0.0, \
                       dev_score))

                test_gold_num, test_predict_num, test_correct_num = \
                    evaluate(test_data, labeler, vocab, config.test_file + '.' + str(global_step))
                test_score = 200.0 * test_correct_num / (test_gold_num + test_predict_num) \
                                if test_correct_num > 0 else 0.0
                print("Test: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
                      (test_correct_num, test_gold_num,  \
                       100.0 * test_correct_num / test_gold_num if test_correct_num > 0 else 0.0, \
                       test_correct_num, test_predict_num, \
                       100.0 * test_correct_num / test_predict_num if test_correct_num > 0 else 0.0, \
                       test_score))

                if dev_score > best_score:
                    print("Exceed best score: history = %.2f, current = %.2f" %(best_score, dev_score))
                    best_score = dev_score
                    if config.save_after > 0 and iter > config.save_after:
                        torch.save(labeler.model.state_dict(), config.save_model_path)


def evaluate(data, labeler, vocab, outputFile):
    start = time.time()
    labeler.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    total_gold_entity_num, total_predict_entity_num, total_correct_entity_num = 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False, False):
        words, extwords, srltags, predicts, inmasks, labels, outmasks = \
            batch_data_variable(onebatch, vocab)
        count = 0
        predict_labels = labeler.label(words, extwords, srltags, predicts, inmasks)
        for result in batch_variable_srl(onebatch, predict_labels, vocab):
            printSRL(output, result)
            gold_entity_num, predict_entity_num, correct_entity_num = evalSRLExact(onebatch[count], result)
            total_gold_entity_num += gold_entity_num
            total_predict_entity_num += predict_entity_num
            total_correct_entity_num += correct_entity_num
            count += 1

    output.close()

    #R = np.float64(total_correct_entity_num) * 100.0 / np.float64(total_gold_entity_num)
    #P = np.float64(total_correct_entity_num) * 100.0 / np.float64(total_predict_entity_num)
    #F = np.float64(total_correct_entity_num) * 200.0 / np.float64(total_gold_entity_num + total_predict_entity_num)


    end = time.time()
    during_time = float(end - start)
    print("sentence num: %d,  parser time = %.2f " % (len(data), during_time))

    return total_gold_entity_num, total_predict_entity_num, total_correct_entity_num


class Optimizer:
    def __init__(self, parameter, config):
        self.optim = torch.optim.Adam(parameter, lr=config.learning_rate, betas=(config.beta_1, config.beta_2),
                                      eps=config.epsilon)
        #self.optim = torch.optim.Adadelta(parameter, lr=1.0, rho=0.95)
        decay, decay_step = config.decay, config.decay_steps
        l = lambda epoch: decay ** (epoch // decay_step)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=l)

    def step(self):
        self.optim.step()
        self.schedule()
        self.optim.zero_grad()

    def schedule(self):
        self.scheduler.step()

    def zero_grad(self):
        self.optim.zero_grad()

    @property
    def lr(self):
        return self.scheduler.get_lr()


if __name__ == '__main__':
    random.seed(666)
    np.random.seed(666)
    torch.cuda.manual_seed(666)
    torch.manual_seed(666)

    ### gpu
    gpu = torch.cuda.is_available()
    print("GPU available: ", gpu)
    print("CuDNN: \n", torch.backends.cudnn.enabled)

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config_file', default='examples/default.cfg')
    argparser.add_argument('--thread', default=4, type=int, help='thread num')
    argparser.add_argument('--use-cuda', action='store_true', default=True)

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)

    vocab = creat_vocab(config.train_file, config.min_occur_count)
    vec = vocab.load_pretrained_embs(config.pretrained_embeddings_file)

    pickle.dump(vocab, open(config.save_vocab_path, 'wb'))

    args, extra_args = argparser.parse_known_args()
    config = Configurable(args.config_file, extra_args)
    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    model = eval(config.model)(vocab, config, vec)
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()

    labeler = SRLLabeler(model)

    data = read_corpus(config.train_file)
    dev_data = read_corpus(config.dev_file)
    test_data = read_corpus(config.test_file)

    train(data, dev_data, test_data, labeler, vocab, config)
