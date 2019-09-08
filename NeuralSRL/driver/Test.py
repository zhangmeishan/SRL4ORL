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

def evaluate(data, labeler, vocab, outputFile):
    start = time.time()
    labeler.model.eval()
    output = open(outputFile, 'w', encoding='utf-8')
    total_gold_entity_num, total_predict_entity_num, total_correct_entity_num = 0, 0, 0

    for onebatch in data_iter(data, config.test_batch_size, False, False):
        words, extwords, predicts, inmasks, labels, outmasks = \
            batch_data_variable(onebatch, vocab)
        count = 0
        predict_labels = labeler.label(words, extwords, predicts, inmasks)
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

    vocab = pickle.load(open(config.load_vocab_path, 'rb'))
    vec = vocab.create_pretrained_embs(config.pretrained_embeddings_file)

    torch.set_num_threads(args.thread)

    config.use_cuda = False
    if gpu and args.use_cuda: config.use_cuda = True
    print("\nGPU using status: ", config.use_cuda)

    # print(config.use_cuda)

    model = eval(config.model)(vocab, config, vec)
    model.load_state_dict(torch.load(config.load_model_path, map_location=lambda storage, loc: storage))
    if config.use_cuda:
        torch.backends.cudnn.enabled = True
        model = model.cuda()

    labeler = SRLLabeler(model)
    test_data = read_corpus(config.test_file)

    gold_num, predict_num, correct_num = \
        evaluate(test_data, labeler, vocab, config.test_file + '.out')
    test_score = 200.0 * correct_num / (gold_num + predict_num) if correct_num > 0 else 0.0
    print("Dev: Recall = %d/%d = %.2f, Precision = %d/%d =%.2f, F-measure = %.2f" % \
          (correct_num, gold_num, 100.0 * correct_num / gold_num if correct_num > 0 else 0.0, \
           correct_num, predict_num, 100.0 * correct_num / predict_num if correct_num > 0 else 0.0, \
           test_score))


