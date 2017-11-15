import tensorflow as tf
from tensorflow.python.ops import tensor_array_ops, control_flow_ops
from tensorflow.python.ops import variable_scope
import numpy as np
from data import gen_vocab2dis_vocab
PAD_TOKEN = "[PAD]"
STOP_DECODING = '[STOP]'


class Rollout(object):
    def __init__(self, generator, update_rate):
        self.generator = generator
        self.update_rate = update_rate
        self._gen_hps = self.generator.hps
        self.g_embeddings = self.generator.embeddings

        #######################################################################
        # placeholder definition
        self.summ = tf.placeholder(
            tf.int32, shape=[self._gen_hps.batch_size, self._gen_hps.max_dec_steps])
        self.cell_c = tf.placeholder(
            tf.float32, shape=[self._gen_hps.batch_size, self._gen_hps.hidden_dim])
        self.cell_h = tf.placeholder(
            tf.float32, shape=[self._gen_hps.batch_size, self._gen_hps.hidden_dim])
        # self.enc_states = tf.placeholder(
        #     tf.float32, shape=[self._gen_hps.batch_size, self._gen_hps.max_enc_steps,
        #                        self._gen_hps.hidden_dim])
        # this should be changed
        init_dec_in_state = tf.contrib.rnn.LSTMStateTuple(self.cell_c, self.cell_h)
        # sequence of tokens generated by generator
        self.given_num = tf.placeholder(tf.int32)

        # processed for batch
        with tf.device("/cpu:0"):
            self.emb_summ = tf.transpose(
                tf.nn.embedding_lookup(self.g_embeddings, self.summ), perm=[1, 0, 2])
            # seq_length x batch_size x emb_dim

        emb_summ_ar = tensor_array_ops.TensorArray(
            dtype=tf.float32, size=self._gen_hps.max_dec_steps)
        emb_summ_ar = emb_summ_ar.unstack(self.emb_summ)

        summ_ar = tensor_array_ops.TensorArray(dtype=tf.int32, size=self._gen_hps.max_dec_steps)
        summ_ar = summ_ar.unstack(tf.transpose(self.summ, perm=[1, 0]))
        ######################################################################

        self.gen_summ_ar = tensor_array_ops.TensorArray(
            dtype=tf.int32, size=self._gen_hps.max_dec_steps, dynamic_size=False, infer_shape=True)

        with tf.variable_scope('rollout_loops'):
            def recurrence_given(i, dec_input, dec_in_state, given_num, gen_summ):
                next_input_id, new_state = self.generator.decode_onestep([dec_input], dec_in_state)
                next_input = emb_summ_ar.read(i)
                gen_summ = gen_summ.write(i, summ_ar.read(i))
                return i+1, next_input, new_state, given_num, gen_summ

            def recurrence_rollout(i, dec_input, dec_in_state, given_num, gen_summ):
                next_input_id, new_state = self.generator.decode_onestep([dec_input], dec_in_state)
                next_input = tf.nn.embedding_lookup(self.g_embeddings, next_input_id)
                gen_summ = gen_summ.write(i, next_input_id)
                return i+1, next_input, new_state, given_num, gen_summ

            i, _, next_input, new_state, given_num, self.gen_summ_ar = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, given_num, _4: i < given_num,
                body=recurrence_given,
                loop_vars=(tf.constant(1, dtype=tf.int32), emb_summ_ar.read(0),
                           init_dec_in_state, self.given_num, self.gen_summ_ar))

            variable_scope.get_variable_scope().reuse_variables()
            # reuse variables between python loops is needed

            _, _, _, _, _, self.gen_summ_ar = control_flow_ops.while_loop(
                cond=lambda i, _1, _2, _3, _4: i < self._gen_hps.max_dec_steps,
                body=recurrence_rollout,
                loop_vars=(i, next_input, new_state, given_num, self.gen_summ_ar))

        self.gen_summ_ar = self.gen_summ_ar.stack()  # seq_length x batch_size
        self.gen_summ_ar = tf.transpose(self.gen_summ_ar, perm=[1, 0])
        self.gen_summ_ar = tf.stop_gradient(self.gen_summ_ar)
        # batch_size x seq_length

    def get_reward(self, sess, gen_vocab, dis_vocab, source_batch,
                   enc_states, dec_in_state, samples, rollout_num, discriminator):
        # dec_in_state is [batch_size, hidden_dim * 2] and that should be
        # changed to [batch_size, hidden_dim] for the attention_decoder
        rewards = []

        article_oovs = source_batch.art_oovs if self._gen_hps.pointer_gen else None
        enc_inputs_words = source_batch.enc_batch_extend_vocab \
            if self._gen_hps.pointer_gen else source_batch.enc_batch
        enc_inputs_chars = gen_vocab2dis_vocab(
            enc_inputs_words, gen_vocab, article_oovs,
            dis_vocab, self._gen_hp.max_enc_steps, PAD_TOKEN)

        for i in range(rollout_num):
            for given_num in range(1, self._gen_hps.max_dec_steps):
                feed_dict = {}
                feed_dict[self.summ] = samples
                feed_dict[self.generator.enc_batch] = source_batch.enc_batch
                # this is the source
                feed_dict[self.generator.enc_lens] = source_batch.enc_lens
                feed_dict[self.given_num] = given_num
                feed_dict[self.generator.enc_states] = enc_states
                feed_dict[self.init_dec_in_state.c] = dec_in_state.c
                feed_dict[self.init_dec_in_state.h] = dec_in_state.h
                if self._gen_hps.pointer_gen:
                    feed_dict[self.generator.enc_batch_extend_vocab] = source_batch.enc_batch_extend_vocab
                    # this is the source
                    feed_dict[self.generator.max_art_oovs] = source_batch.max_art_oovs
                # how to deal with the coverage?

                # the unique feature for the pointer gen is the
                # enc_batch_extend_vocab and the max_art_oovs
                # self._enc_batch_extend_vocab = batch.enc_batch_extend_vocab
                # self._max_art_oovs = batch.max_art_oovs
                rollout_samples_words = sess.run(self.gen_summ_ar, feed_dict)
                # how about multiple generators for one discriminator?
                rollout_samples_chars = gen_vocab2dis_vocab(
                    rollout_samples_words, gen_vocab, article_oovs,
                    dis_vocab, self._gen_hps.max_dec_steps, STOP_DECODING)

                feed = {
                    discriminator.dis_input_x: rollout_samples_chars,
                    discriminator.dis_input_xs: enc_inputs_chars,
                    discriminator.dropout_keep_prob: 1.0}
                ypred_for_auc = sess.run(discriminator.dis_ypred_for_auc, feed)
                ypred = np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[given_num - 1] += ypred

            # the last token reward
            samples_chars = gen_vocab2dis_vocab(
                samples, gen_vocab, article_oovs,
                dis_vocab, self._gen_hps.max_dec_steps, STOP_DECODING)
            feed = {
                discriminator.dis_input_x: samples_chars,
                discriminator.dis_input_xs: enc_inputs_chars,
                discriminator.dropout_keep_prob: 1.0}
            ypred_for_auc = sess.run(discriminator.dis_ypred_for_auc, feed)
            ypred = np.array([item[1] for item in ypred_for_auc])
            if i == 0:
                rewards.append(ypred)
            else:
                rewards[self._gen_hps.max_dec_steps - 1] += ypred

        rewards = np.transpose(np.array(rewards)) / (1.0 * rollout_num)
        # batch_size x seq_length
        return rewards
