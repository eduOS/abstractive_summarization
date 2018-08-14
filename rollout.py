from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division
import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
# from tensorflow.python.ops import variable_scope
import numpy as np
from data import strip_pads
from gan_utils import rouge_l
from data import outputsids2words
from data import PAD_TOKEN
from data import START_DECODING # noqa
from data import STOP_DECODING

DEBUG = False
if DEBUG:
    from termcolor import colored # noqa


class Rollout(object):
    def __init__(self, generator, update_rate, decoder_scope):
        self.generator = generator
        self.update_rate = update_rate
        # TODO: for the variables update
        self._gen_hps = self.generator.hps
        self.g_embeddings = self.generator.dec_embeddings
        max_dec_steps = self._gen_hps.max_dec_steps

        #######################################################################
        self.given_num = tf.placeholder(tf.int32, name="given_num")
        self.sample = tf.placeholder(
            tf.int32, shape=[self._gen_hps.batch_size, max_dec_steps+1], name="sample")

        sample_ar = tensor_array_ops.TensorArray(dtype=tf.int32, size=max_dec_steps+1)
        self.sample_ar = sample_ar.unstack(tf.transpose(self.sample, [1, 0]))

        rollout_sample_ar = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=1, dynamic_size=True, infer_shape=True, clear_after_read=False)
        ######################################################################

        with tf.variable_scope(decoder_scope, reuse=True):

            def recurrence_given(i, init_start_ar):
                init_start_ar = init_start_ar.write(i, self.sample_ar.read(i))
                return i+1, init_start_ar

            def recurrence_rollout(i, dec_input):
                dec_input_emb = tf.nn.embedding_lookup(
                    self.g_embeddings, tf.transpose(dec_input.stack(), [1, 0]))
                output_id = self.generator.decode_onestep(dec_input_emb)
                next_input = dec_input.write(i, output_id)
                return i+1, next_input

            j, init_start = control_flow_ops.while_loop(
                cond=lambda i, _1: i < self.given_num,
                body=recurrence_given, loop_vars=(0, rollout_sample_ar))

            _, self.rollout_sample_ar = control_flow_ops.while_loop(
                cond=lambda i, _1: i < max_dec_steps+1,
                body=recurrence_rollout, loop_vars=(j, init_start))

        self.rollout_samples = tf.slice(tf.transpose(self.rollout_sample_ar.stack()), [0, 1], [-1, -1])
        self.rollout_samples_emb = tf.nn.embedding_lookup(self.g_embeddings, self.rollout_samples)

    def get_reward(self, hps_gan, sess, dec_vocab, source_batch, enc_states, k_samples, discriminator):
        rollout_num = hps_gan.rollout_num
        rouge_ratio = hps_gan.rouge_reward_ratio
        dis_ratio = hps_gan.dis_reward_ratio
        max_dec_steps = self._gen_hps.max_dec_steps
        weights = map(lambda i: 0.8**i, range([max_dec_steps - hps_gan.rollout_start + 1]))

        articles = source_batch.enc_batch
        article_lens = source_batch.enc_lens
        # batch_size = int(articles.shape[0])
        stop_token = dec_vocab.word2id(STOP_DECODING)
        emb_articles = sess.run(
            self.generator.enc_temp_embedded,
            feed_dict={self.generator.enc_temp_batch: articles})

        k_rewards = []

        for k, samples in enumerate(k_samples):
            dis_rewards = []
            rouge_rewards = []
            no_stop_samples = strip_pads(
                samples.tolist(),
                stop_token,
                keep_length=True,
                PAD_ID=dec_vocab.word2id(PAD_TOKEN))
            for ir in range(rollout_num):
                for given_num in range(hps_gan.rollout_start, max_dec_steps+1):
                    self.sample_emb_ls = []

                    feed_dict = {}
                    feed_dict[self.sample] = samples
                    feed_dict[self.given_num] = given_num
                    feed_dict[self.generator.enc_padding_mask] = source_batch.enc_padding_mask
                    feed_dict[self.generator.attention_keys] = enc_states
                    feed_dict[self.generator.emb_enc_inputs] = emb_articles

                    rollout_samples = sess.run([self.rollout_samples], feed_dict)
                    # how about multiple generators for one discriminator?
                    if dis_ratio:
                        rollout_samples_batch = strip_pads(
                            rollout_samples.tolist(),
                            stop_token,
                            keep_length=True,
                            PAD_ID=dec_vocab.word2id(PAD_TOKEN))

                        emb_rollout_samples = sess.run(
                            self.generator.dec_temp_embedded,
                            feed_dict={self.generator.dec_temp_batch: rollout_samples_batch})

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
                        rouge_scores = []
                        summaries = outputsids2words(strip_pads(rollout_samples.tolist(), stop_token), dec_vocab)
                        references = source_batch.original_abstracts
                        for s, r in zip(summaries, references):
                            rouge = rouge_l(s, r.split())
                            if DEBUG:
                                print(r)
                                print(colored(' '.join(s[:given_num]), 'green') + " " + colored(' '.join(s[given_num:]), 'red'))
                                print()
                            rouge_scores.append(rouge)
                        if ir == 0:
                            rouge_rewards.append(np.array(rouge_scores))
                        else:
                            rouge_rewards[given_num-1] += np.array(rouge_scores)

                if dis_ratio:
                    emb_samples = sess.run(
                        self.generator.dec_temp_embedded,
                        feed_dict={self.generator.dec_temp_batch: no_stop_samples})

                    # the last token reward
                    feed = {
                        discriminator.inputs: emb_samples[:, 1:, :],
                        discriminator.conditions: emb_articles,
                        discriminator.condition_lens: article_lens
                    }
                    ypred_for_auc = sess.run(discriminator.dis_ypred_for_auc, feed)
                    if ir == 0:
                        dis_rewards.append(ypred_for_auc)
                    else:
                        dis_rewards[max_dec_steps-1] += ypred_for_auc

                if rouge_ratio:
                    rouge_scores = []
                    summaries = outputsids2words(strip_pads(samples.tolist(), stop_token), dec_vocab)
                    references = source_batch.original_abstracts
                    for s, r in zip(summaries, references):
                        rouge = rouge_l(s, r.split())
                        rouge_scores.append(rouge)
                    if ir == 0:
                        rouge_rewards.append(np.array(rouge_scores))
                    else:
                        rouge_rewards[max_dec_steps-1] += np.array(rouge_scores)

            if rouge_ratio:
                rouge_rewards = np.transpose(np.array(rouge_rewards))
                if hps_gan.subtract:
                    rouge_rewards = rouge_rewards[:, 1:] - rouge_rewards[:, :-1]
                else:
                    rouge_rewards = rouge_rewards[:, 1:]

            if dis_ratio:
                dis_rewards = np.transpose(np.array(dis_rewards))
                if hps_gan.subtract:
                    dis_rewards = dis_rewards[:, 1:] - dis_rewards[:, :-1]
                else:
                    dis_rewards = dis_rewards[:, 1:]

            if rouge_ratio == 1:
                rewards = rouge_rewards
            elif rouge_ratio == 0:
                rewards = dis_rewards
            else:
                rewards = (1 - rouge_ratio)*dis_rewards + rouge_ratio*rouge_rewards

            average_rewards = rewards / (1.0 * rollout_num) * weights
            k_rewards.append(average_rewards)

        return k_rewards
