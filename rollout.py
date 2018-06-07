from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
# from tensorflow.python.ops import variable_scope
import numpy as np
from data import strip_pads
import data
from gan_utils import rouge_l
PAD_TOKEN = "[PAD]"
START_DECODING = '[START]'
STOP_DECODING = '[STOP]'


class Rollout(object):
    def __init__(self, generator, update_rate, decoder_scope):
        self.generator = generator
        self.update_rate = update_rate
        # TODO: for the variables update
        self._gen_hps = self.generator.hps
        self.g_embeddings = self.generator.embeddings
        max_dec_steps = self._gen_hps.max_dec_steps
        #######################################################################

        self.sample = tf.placeholder(
            tf.int32, shape=[self._gen_hps.batch_size, max_dec_steps], name="sample")
        init_start, _ = tf.split(self.sample, [self.given_num, max_dec_steps-self.given_num], axis=1)
        self.emb_sample = tf.nn.embedding_lookup(self.g_embeddings, self.sample)
        self.given_num = tf.placeholder(tf.int32, name="given_num")
        init_start_emb, _ = tf.split(self.emb_sample, [self.given_num, max_dec_steps-self.given_num], axis=1)

        ######################################################################

        self.rollout_sample_ar = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=max_dec_steps, dynamic_size=False, infer_shape=True)

        with tf.variable_scope(decoder_scope, reuse=True):

            def recurrence_rollout(i, dec_input, rollout_sample_ar):
                output_id = self.generator.decode_onestep(dec_input)
                rollout_sample_ar = rollout_sample_ar.write(i, output_id)
                next_input = tf.concat([rollout_sample_ar.stack(), output_id], axis=1)
                next_input_emb = tf.nn.embedding_lookup(self.g_embeddings, next_input)
                return i+1, next_input_emb, rollout_sample_ar

            _, _, self.rollout_sample_ar = control_flow_ops.while_loop(
                cond=lambda i, _1, _2: i < max_dec_steps-self.given_num,
                body=recurrence_rollout,
                loop_vars=(0, init_start_emb, self.rollout_sample_ar))

        self.rollout_sample = tf.concat([init_start, self.rollout_sample_ar.stack()], axis=1)  # seq_length x batch_size

    def get_reward(self, hps_gan, sess, gen_vocab, dis_vocab, source_batch,
                   enc_states, dec_in_state, k_samples, discriminator):
        # dec_in_state is [batch_size, hidden_dim * 2] and that should be
        # changed to [batch_size, hidden_dim] for the attention_decoder
        rollout_num = hps_gan.rollout_num
        rouge_ratio = hps_gan.rouge_reward_ratio
        dis_ratio = hps_gan.dis_reward_ratio
        max_dec_steps = self._gen_hps.max_dec_steps

        articles_extend = source_batch.enc_batch_extend_vocab
        articles = source_batch.enc_batch
        article_lens = source_batch.enc_lens
        batch_size = int(articles_extend.shape[0])
        emb_articles = sess.run(
            self.generator.temp_embedded_seq,
            feed_dict={self.generator.temp_batch: articles})

        k_rewards = []

        for k, samples in enumerate(k_samples):
            dis_rewards = []
            rouge_rewards = np.zeros((max_dec_steps+1, batch_size))
            for ir in range(rollout_num):
                for given_num in range(hps_gan.rollout_start, max_dec_steps):

                    feed_dict = {}
                    feed_dict[self.sample] = samples
                    # this is the source
                    # feed_dict[self.generator.enc_lens] = source_batch.enc_lens
                    feed_dict[self.given_num] = given_num
                    feed_dict[self.generator.enc_states] = enc_states
                    feed_dict[self.generator.enc_padding_mask] = source_batch.enc_padding_mask
                    feed_dict[self.cell_c] = dec_in_state.c
                    feed_dict[self.cell_h] = dec_in_state.h
                    feed_dict[self.generator.enc_batch_extend_vocab] = articles_extend
                    feed_dict[self.generator.max_art_oovs] = source_batch.max_art_oovs
                    # how to deal with the coverage?

                    # the unique feature for the pointer gen is the
                    # enc_batch_extend_vocab and the max_art_oovs
                    rollout_samples_extend = sess.run(self.rollout_sample, feed_dict)
                    rollout_samples = np.where(
                        np.less(samples, self._gen_hps.gen_vocab_size),
                        rollout_samples_extend, np.array(
                            [[gen_vocab.word2id(data.UNKNOWN_TOKEN)] * max_dec_steps
                             ] * self._gen_hps.batch_size))
                    # how about multiple generators for one discriminator?
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
                        rpred = rouge_l(strip_pads(rollout_samples_extend.tolist(), gen_vocab.word2id(STOP_DECODING)),
                                        strip_pads(source_batch.dec_batch.tolist(), gen_vocab.word2id(PAD_TOKEN)), beta=0.5)
                        rouge_rewards[given_num] += np.array(rpred)

                if dis_ratio:
                    emb_samples = sess.run(
                        self.generator.temp_embedded_seq,
                        feed_dict={self.generator.temp_batch: samples})

                    # the last token reward
                    feed = {
                        discriminator.inputs: emb_samples,
                        discriminator.conditions: emb_articles,
                        discriminator.condition_lens: article_lens
                    }
                    ypred_for_auc = sess.run(discriminator.dis_ypred_for_auc, feed)
                    if ir == 0:
                        dis_rewards.append(ypred_for_auc)
                    else:
                        dis_rewards[max_dec_steps-1] += ypred_for_auc
                if rouge_ratio:
                    rpred = rouge_l(strip_pads(samples.tolist(), gen_vocab.word2id(STOP_DECODING)),
                                    strip_pads(source_batch.dec_batch.tolist(), gen_vocab.word2id(PAD_TOKEN)), beta=0.5)
                    rouge_rewards[max_dec_steps] += np.array(rpred)

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
