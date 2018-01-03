from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
# from tensorflow.python.ops import variable_scope
import numpy as np
from data import gen_vocab2dis_vocab
import data
PAD_TOKEN = "[PAD]"
STOP_DECODING = '[STOP]'
FLAGS = tf.app.flags.FLAGS


class Rollout(object):
    def __init__(self, generator, update_rate, decoder_scope):
        pass

    def roll_out():
        return samples

    def get_reward(self, sess, gen_vocab, dis_vocab, source_batch,
                   enc_states, dec_in_state, samples,
                   rollout_num, discriminator):
        # dec_in_state is [batch_size, hidden_dim * 2] and that should be
        # changed to [batch_size, hidden_dim] for the attention_decoder
        rewards = []

        article_oovs = source_batch.art_oovs
        art_words = source_batch.enc_batch_extend_vocab
        art_chars = gen_vocab2dis_vocab(
            art_words, gen_vocab, article_oovs,
            dis_vocab, discriminator.hps.max_enc_steps, PAD_TOKEN)
        # abs_chars = np.array(gen_vocab2dis_vocab(
        #     source_batch.target_batch, gen_vocab, article_oovs,
        #     dis_vocab, self._gen_hps.max_dec_steps, STOP_DECODING))

        for i in range(rollout_num):
            for given_num in range(2, self._gen_hps.max_dec_steps+1):
                feed_dict = {}
                feed_dict[self.summ] = samples
                # this is the source
                feed_dict[self.generator.enc_lens] = source_batch.enc_lens
                feed_dict[self.given_num] = given_num
                feed_dict[self.generator.enc_states] = enc_states
                feed_dict[self.generator.enc_padding_mask] = source_batch.enc_padding_mask
                feed_dict[self.cell_c] = dec_in_state.c
                feed_dict[self.cell_h] = dec_in_state.h
                feed_dict[self.generator.enc_batch_extend_vocab] = art_words
                # this is the source
                feed_dict[self.generator.max_art_oovs] = source_batch.max_art_oovs
                # how to deal with the coverage?

                # the unique feature for the pointer gen is the
                # enc_batch_extend_vocab and the max_art_oovs
                rollout_samples_words = sess.run(self.gen_summ_ar, feed_dict)
                # how about multiple generators for one discriminator?
                rollout_samples_chars = gen_vocab2dis_vocab(
                    rollout_samples_words, gen_vocab, article_oovs,
                    dis_vocab, discriminator.hps.max_dec_steps, STOP_DECODING, art_words, print_sample=False)

                feed = {
                    discriminator.inputs: rollout_samples_chars,
                    discriminator.conditions: art_chars}
                ypred_for_auc = sess.run(discriminator.dis_ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 2] += ypred

            # the last token reward
            samples_without_start = [s[1:] for s in samples]
            samples_chars = gen_vocab2dis_vocab(
                samples_without_start, gen_vocab, article_oovs,
                dis_vocab, discriminator.hps.max_dec_steps, STOP_DECODING)
            feed = {
                discriminator.inputs: samples_chars,
                discriminator.conditions: art_chars}
            ypred_for_auc = sess.run(discriminator.dis_ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self._gen_hps.max_dec_steps - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)
        # batch_size x seq_length
        return rewards
