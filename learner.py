from nn.utils.config_factory import config
from nn.utils.generic_utils import *

import torch
import sys
import os
import time

import decoder
import evaluation
from dataset import *
import config

from util import trace_back


def second2minute(time1, time2):
    return round((time1 - time2) / 60, 3)


class Learner(object):
    def __init__(self, model, train_data, val_data=None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

        logging.info('initial learner with training set [%s] (%d examples)',
                     train_data.name,
                     train_data.count)
        if val_data:
            logging.info('validation set [%s] (%d examples)', val_data.name, val_data.count)

    def train(self, optimizer):
        dataset = self.train_data
        nb_train_sample = dataset.count
        index_array = np.arange(nb_train_sample)

        nb_epoch = config.max_epoch
        batch_size = config.batch_size

        logging.info('begin training')
        cum_updates = 0
        patience_counter = 0
        early_stop = False
        history_valid_perf = []
        history_valid_bleu = []
        history_valid_acc = []
        history_losses = []
        best_model_params = best_model_by_acc = best_model_by_bleu = None

        # train_data_iter = DataIterator(self.train_data, batch_size)
        original_start = time.time()
        start_time = original_start

        self.model.train()
        for epoch in range(nb_epoch):
            # train_data_iter.reset()
            # if shuffle:
            np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)

            # epoch begin
            sys.stdout.write('Epoch %d' % epoch)
            begin_time = time.time()
            cum_nb_examples = 0
            loss = 0.0

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                optimizer.zero_grad()
                cum_updates += 1

                batch_ids = index_array[batch_start:batch_end]
                examples = dataset.get_examples(batch_ids)
                cur_batch_size = len(examples)

                inputs = dataset.get_prob_func_inputs(batch_ids)

                if not config.enable_copy:
                    tgt_action_seq = inputs[1]
                    tgt_action_seq_type = inputs[2]

                    for i in xrange(cur_batch_size):
                        for t in xrange(tgt_action_seq[i].shape[0]):
                            if tgt_action_seq_type[i, t, 2] == 1:
                                # can only be copied
                                if tgt_action_seq_type[i, t, 1] == 0:
                                    tgt_action_seq_type[i, t, 1] = 1
                                    tgt_action_seq[i, t, 1] = 1  # index of <unk>

                                tgt_action_seq_type[i, t, 2] = 0

                encode_outputs = self.model(*inputs)
                batch_loss = encode_outputs.item()
                # print("=" * 60)
                # trace_back(encode_outputs.grad_fn)
                # print("=" * 60)
                encode_outputs.backward()
                optimizer.step()
                logging.debug('prob_func finished computing')

                cum_nb_examples += cur_batch_size
                loss += batch_loss * batch_size

                logging.debug('Batch %d, avg. loss = %f', batch_index, batch_loss)

                if batch_index == 4:
                    elapsed = time.time() - begin_time
                    eta = nb_train_sample / (cum_nb_examples / elapsed)
                    print(', eta %ds' % eta)
                    sys.stdout.flush()

                if cum_updates % config.valid_per_batch == 0:
                # if cum_updates % 40 == 0:
                    logging.info('begin validation')

                    if config.data_type == 'ifttt':
                        decode_results = decoder.decode_ifttt_dataset(self.model, self.val_data, verbose=False)
                        channel_acc, channel_func_acc, prod_f1 = evaluation.evaluate_ifttt_results(self.val_data, decode_results, verbose=False)

                        val_perf = channel_func_acc
                        logging.info('channel accuracy: %f', channel_acc)
                        logging.info('channel+func accuracy: %f', channel_func_acc)
                        logging.info('prod F1: %f', prod_f1)
                    else:
                        decode_results = decoder.decode_python_dataset(self.model, train_data=dataset, test_data=self.val_data, verbose=True)
                        bleu, accuracy = evaluation.evaluate_decode_results(self.val_data, decode_results, verbose=True)

                        val_perf = eval(config.valid_metric)

                        logging.info('avg. example bleu: %f', bleu)
                        logging.info('accuracy: %f', accuracy)

                        if len(history_valid_acc) == 0 or accuracy > np.array(history_valid_acc).max():
                            best_model_by_acc = self.model.state_dict()
                            # logging.info('current model has best accuracy')
                        history_valid_acc.append(accuracy)

                        if len(history_valid_bleu) == 0 or bleu > np.array(history_valid_bleu).max():
                            best_model_by_bleu = self.model.state_dict()
                            # logging.info('current model has best accuracy')
                        history_valid_bleu.append(bleu)

                    if len(history_valid_perf) == 0 or val_perf > np.array(history_valid_perf).max():
                        best_model = self.model.state_dict()
                        patience_counter = 0
                        logging.info('save current best model')
                        torch.save(self.model.state_dict(), os.path.join(config.output_dir, 'model.npz'))
                    else:
                        patience_counter += 1
                        logging.info('hitting patience_counter: %d', patience_counter)
                        if patience_counter >= config.train_patience:
                            logging.info('Early Stop!')
                            early_stop = True
                            break
                    history_valid_perf.append(val_perf)

                if cum_updates % config.save_per_batch == 0:
                    torch.save(self.model.state_dict(), os.path.join(config.output_dir, 'model.iter%d' % cum_updates))

            logging.info('[Epoch %d] cumulative loss = %f, (took %ds, total %f min) ',
                         epoch,
                         loss / cum_nb_examples,
                         time.time() - begin_time, second2minute(time.time(), original_start))
            history_losses.append(loss / cum_nb_examples)

            if early_stop:
                break

        logging.info('training finished, save the best model')
        # torch.save(best_model.state_dict(), os.path.join(config.output_dir, 'model.npz'))

        try:
            with open('valid_perf.txt', 'w') as fp:
                for x in history_valid_perf:
                    fp.write(x)
                    fp.write('\n')
            fp.close()

            with open('valid_bleu.txt', 'w') as fp:
                for x in history_valid_bleu :
                    fp.write(x)
                    fp.write('\n')
            fp.close()

            with open('valid_acc.txt', 'w') as fp:
                for x in history_valid_acc:
                    fp.write(x)
                    fp.write('\n')
            fp.close()

            with open('train_loss.txt', 'w') as fp:
                for x in history_losses:
                    fp.write(x)
                    fp.write('\n')
            fp.close()
        except:
            print "Fail to save result"
            pass

        if config.data_type == 'django' or config.data_type == 'hs':
            logging.info('save the best model by accuracy')
            torch.save(best_model_by_acc, os.path.join(config.output_dir, 'model.best_acc.npz'))

            logging.info('save the best model by bleu')
            torch.save(best_model_by_bleu, os.path.join(config.output_dir, 'model.best_bleu.npz'))


class DataIterator:
    def __init__(self, dataset, batch_size=10):
        self.dataset = dataset
        self.batch_size = batch_size
        self.index_array = np.arange(self.dataset.count)
        self.ptr = 0
        self.buffer_size = batch_size * 5
        self.buffer = []

    def reset(self):
        self.ptr = 0
        self.buffer = []
        np.random.shuffle(self.index_array)

    def __iter__(self):
        return self

    def next_batch(self):
        batch = self.buffer[:self.batch_size]
        del self.buffer[:self.batch_size]

        batch_ids = [e.eid for e in batch]

        return batch, batch_ids

    def next(self):
        if self.buffer:
            return self.next_batch()
        else:
            if self.ptr >= self.dataset.count:
                raise StopIteration

            self.buffer = self.index_array[self.ptr:self.ptr + self.buffer_size]

            # sort buffer contents
            examples = self.dataset.get_examples(self.buffer)
            self.buffer = sorted(examples, key=lambda e: len(e.actions))

            self.ptr += self.buffer_size

            return self.next_batch()