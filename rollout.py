from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
# from tensorflow.python.ops import variable_scope
import numpy as np
from data import strip_pads
import time
import data
from check_rouge import calc_rouge
PAD_TOKEN = "[PAD]"
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'


class Rollout(object):
    def __init__(self, generator, update_rate, decoder_scope):
        self.generator = generator
        self.update_rate = update_rate
        self._gen_hps = self.generator.hps
        self.g_embeddings = self.generator.dec_embeddings
        start_tokens = np.array([self.generator.dec_vocab.word2id(data.START_DECODING)] * self._gen_hps.batch_size)
        emb_start_token = tf.nn.embedding_lookup(self.g_embeddings, start_tokens)
        next_input = emb_start_token

        self.sample = tf.placeholder(
            tf.int32, shape=[self._gen_hps.batch_size, self._gen_hps.max_dec_steps], name="sample")
        self.cell_c = tf.placeholder(
            tf.float32, shape=[self._gen_hps.batch_size, self._gen_hps.hidden_dim], name="cell_c")
        self.cell_h = tf.placeholder(
            tf.float32, shape=[self._gen_hps.batch_size, self._gen_hps.hidden_dim], name="cell_h")
        self.given_num = tf.placeholder(tf.int32, name="given_num")
        init_dec_in_state = tf.contrib.rnn.LSTMStateTuple(self.cell_c, self.cell_h)
        new_state = init_dec_in_state

        self.emb_sample = tf.transpose(
            tf.nn.embedding_lookup(self.g_embeddings, self.sample), perm=[1, 0, 2])

        emb_sample_ar = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self._gen_hps.max_dec_steps)
        emb_sample_ar = emb_sample_ar.unstack(self.emb_sample)

        sample_ar = tensor_array_ops.TensorArray(dtype=tf.int32, size=self._gen_hps.max_dec_steps)
        sample_ar = sample_ar.unstack(tf.transpose(self.sample, perm=[1, 0]))

        self.rollout_sample_ar = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=self._gen_hps.max_dec_steps, dynamic_size=False, infer_shape=True)

        with tf.variable_scope(decoder_scope, reuse=True):
            def recurrence_given(i, dec_input, dec_in_state, given_num, rollout_sample_ar):
                next_input_id, new_state = self.generator.decode_onestep([dec_input], dec_in_state)
                emb_next_input = emb_sample_ar.read(i)
                rollout_sample_ar = rollout_sample_ar.write(i, sample_ar.read(i))
                return i+1, emb_next_input, new_state, given_num, rollout_sample_ar

            i, next_input, new_state, given_num, self.rollout_sample_ar = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, given_num, _4: i < given_num,
                body=recurrence_given,
                loop_vars=(tf.constant(0, dtype=tf.int32), next_input,
                           new_state, self.given_num, self.rollout_sample_ar))

            def recurrence_rollout(i, dec_input, dec_in_state, rollout_sample_ar):
                output_id, new_state = self.generator.decode_onestep([dec_input], dec_in_state)
                rollout_sample_ar = rollout_sample_ar.write(i, output_id)
                next_input_emb = tf.nn.embedding_lookup(self.g_embeddings, output_id)
                return i+1, next_input_emb, new_state, rollout_sample_ar

            _, _, _, self.rollout_sample_ar = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, _3: i < self._gen_hps.max_dec_steps,
                body=recurrence_rollout,
                loop_vars=(i, next_input, new_state, self.rollout_sample_ar))

        self.rollout_sample = self.rollout_sample_ar.stack()  # seq_length x batch_size
        self.rollout_sample = tf.transpose(self.rollout_sample, perm=[1, 0])

    def get_reward(self, hps_gan, sess, gen_vocab, source_batch,
                   attention_keys, dec_in_state, k_samples, discriminator):
        rollout_num = hps_gan.rollout_num
        rouge_ratio = hps_gan.rouge_reward_ratio
        dis_ratio = hps_gan.dis_reward_ratio

        articles = source_batch.enc_batch
        article_lens = source_batch.enc_lens
        batch_size = int(articles.shape[0])
        emb_articles = sess.run(
            self.generator.temp_embedded_seq,
            feed_dict={self.generator.temp_batch: articles})

        k_rewards = []

        for k, samples in enumerate(k_samples):
            dis_rewards = []
            rouge_rewards = np.zeros((self._gen_hps.max_dec_steps+1, batch_size))
            for ir in range(rollout_num):
                for given_num in range(hps_gan.rollout_start, self._gen_hps.max_dec_steps):

                    feed_dict = {}
                    feed_dict[self.sample] = samples
                    feed_dict[self.given_num] = given_num
                    feed_dict[self.generator.attention_keys] = attention_keys
                    feed_dict[self.generator.enc_padding_mask] = source_batch.enc_padding_mask
                    feed_dict[self.cell_c] = dec_in_state.c
                    feed_dict[self.cell_h] = dec_in_state.h

                    rollout_samples = sess.run(self.rollout_sample, feed_dict)
                    if dis_ratio:
                        emb_rollout_samples = sess.run(
                            self.generator.temp_embedded_seq,
                            feed_dict={self.generator.temp_batch: rollout_samples})

                        feed = {
                            discriminator.inputs: emb_rollout_samples,
                            discriminator.conditions: emb_articles,
                            discriminator.condition_lens: article_lens}
                        ypred_for_auc = sess.run(discriminator.dis_ypred_for_auc, feed)
                        if ir == 0:
                            dis_rewards.append(ypred_for_auc)
                        else:
                            dis_rewards[given_num-1] += ypred_for_auc

                    if rouge_ratio:
                        _outputs = [" ".join(s) for s in data.outputsids2words(strip_pads(rollout_samples.tolist(), gen_vocab.word2id(STOP_DECODING)), self._vocab)]
                        _reference = [" ".join(s) for s in data.outputsids2words(strip_pads(source_batch.dec_batch.tolist(), gen_vocab.word2id(PAD_TOKEN)), self._vocab)]

                        _, _, rpred = calc_rouge(_outputs, _reference)
                        rouge_rewards[given_num] += np.array(rpred)

                if dis_ratio:
                    emb_samples = sess.run(
                        self.generator.temp_embedded_seq,
                        feed_dict={self.generator.temp_batch: samples})

                    feed = {
                        discriminator.inputs: emb_samples,
                        discriminator.conditions: emb_articles,
                        discriminator.condition_lens: article_lens
                    }
                    ypred_for_auc = sess.run(discriminator.dis_ypred_for_auc, feed)
                    if ir == 0:
                        dis_rewards.append(ypred_for_auc)
                    else:
                        dis_rewards[self._gen_hps.max_dec_steps-1] += ypred_for_auc
                if rouge_ratio:
                    _outputs = [" ".join(s) for s in data.outputsids2words(strip_pads(samples.tolist(), gen_vocab.word2id(STOP_DECODING)), self._vocab)]
                    _reference = [" ".join(s) for s in data.outputsids2words(strip_pads(source_batch.dec_batch.tolist(), gen_vocab.word2id(PAD_TOKEN)), self._vocab)]

                    _, _, rpred = calc_rouge(_outputs, _reference)
                    rouge_rewards[self._gen_hps.max_dec_steps] += np.array(rpred)

            if rouge_ratio:
                rouge_rewards = np.transpose(rouge_rewards)
                rouge_rewards = rouge_rewards[:, 1:] - rouge_rewards[:, :-1]

            if dis_ratio:
                dis_rewards = np.transpose(np.array(dis_rewards))
                dis_rewards = dis_rewards[:, 1:] - dis_rewards[:, :-1]

            if rouge_ratio == 1:
                rewards = rouge_rewards
            elif rouge_ratio == 0:
                rewards = dis_rewards
            else:
                rewards = (1 - rouge_ratio)*dis_rewards + rouge_ratio*rouge_rewards

            average_rewards = rewards / (1.0 * rollout_num)
            k_rewards.append(average_rewards)

        return k_rewards
