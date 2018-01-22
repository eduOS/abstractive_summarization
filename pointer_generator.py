# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the
sequence-to-sequence model"""
from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import sys
import time
import numpy as np
import tensorflow as tf
from termcolor import colored
from attention_decoder import attention_decoder
# from share_function import tableLookup
from six.moves import xrange

FLAGS = tf.app.flags.FLAGS


class PointerGenerator(object):
    """A class to represent a sequence-to-sequence model for text summarization.
    Supports both baseline mode, pointer-generator mode, and coverage"""

    def __init__(self, hps, vocab):
        self.hps = hps
        self._vocab = vocab

    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input
        data."""
        hps = self.hps
        batch_size = None
        if hps.mode in ["decode", "train_gan"]:
            max_dec_steps = 1
        else:
            max_dec_steps = hps.max_dec_steps

        # -------- placeholders for training generatror and beam search decoding
        # encoder part
        self.enc_batch = tf.placeholder(tf.int32, [batch_size, None], name='enc_batch')
        self.enc_lens = tf.placeholder(tf.int32, [batch_size], name='enc_lens')
        self.enc_padding_mask = tf.placeholder(tf.float32, [batch_size, None], name='enc_padding_mask')
        self.enc_batch_extend_vocab = tf.placeholder(tf.int32, [batch_size, None], name='enc_batch_extend_vocab')
        self.max_art_oovs = tf.placeholder(tf.int32, [], name='max_art_oovs')

        # decoder part
        # when max_dec_steps is 1, this is for beam search decoding, it is for
        # getting the loss otherwise
        self._dec_batch = tf.placeholder(tf.int32, [batch_size, max_dec_steps], name='dec_batch')
        self.target_batch = tf.placeholder(tf.int32, [batch_size, hps.max_dec_steps], name='target_batch')
        self.dec_padding_mask = tf.placeholder(tf.float32, [batch_size, hps.max_dec_steps], name='decoder_padding_mask')

        # ------------------------ placeholders for training GAN
        # encoder part
        self.cell_c = tf.placeholder(
            tf.float32, shape=[batch_size, self.hps.hidden_dim])
        self.cell_h = tf.placeholder(
            tf.float32, shape=[batch_size, self.hps.hidden_dim])

        # decoder part
        # in gan training all three modes of the generator are used:
        # decoding for the generating process; training for the tuning and
        # evaluation for its evaluation
        self.k_sample_targets = tf.placeholder(tf.int32, [FLAGS.sample_num, batch_size, hps.max_dec_steps], name='k_sample_targets')
        self.k_sample_targets_mask = tf.placeholder(tf.float32, [FLAGS.sample_num, batch_size, hps.max_dec_steps], name='k_padding_mask_of_the_sample_targets')
        self.k_samples = tf.placeholder(tf.int32, [FLAGS.sample_num, batch_size, hps.max_dec_steps], name='k_samples')
        self.k_rewards = tf.placeholder(tf.float32, shape=[FLAGS.sample_num, batch_size, hps.max_dec_steps], name="k_rewards")

        # ------------------------ placeholders for evaluation
        # decoder part
        self._eval_dec_batch = tf.placeholder(tf.int32, [batch_size, hps.max_dec_steps], name='eval_dec_batch')

        if hps.mode in ["decode", 'train_gan'] and hps.coverage:
            self.prev_coverage = tf.placeholder(tf.float32, [None, None], name='prev_coverage')
            # so this need not to be reloaded and taken gradient hps

    def _make_feed_dict(self, batch, just_enc=False, gan_eval=False, gan=False):
        """Make a feed dictionary mapping parts of the batch to the appropriate
        placeholders.

        Args:
          batch: Batch object
          just_enc: Boolean. If True, only feed the parts needed for the
          encoder.
          update: only for the evaluation and training of the generator in gan training
        """
        if gan_eval:
            gan = True
        feed_dict = {}
        feed_dict[self.enc_batch] = batch.enc_batch
        feed_dict[self.enc_lens] = batch.enc_lens
        feed_dict[self.enc_padding_mask] = batch.enc_padding_mask
        if not just_enc:
            feed_dict[self.enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
            feed_dict[self.max_art_oovs] = batch.max_art_oovs
            feed_dict[self.target_batch] = batch.target_batch
            feed_dict[self.dec_padding_mask] = batch.dec_padding_mask
            if gan_eval:
                feed_dict[self._eval_dec_batch] = batch.dec_batch
            elif not gan:
                feed_dict[self._dec_batch] = batch.dec_batch
        return feed_dict

    def _add_encoder(self, encoder_inputs, seq_len):
        """Add a single-layer bidirectional LSTM encoder to the graph.

        Args:
          encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps,
          emb_size].
          seq_len: Lengths of encoder_inputs (before padding). A tensor of shape
          [batch_size].

        Returns:
          encoder_outputs:
            A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's
            2*hidden_dim because it's the concatenation of the forwards and
            backwards states.
          fw_state, bw_state:
            Each are LSTMStateTuples of shape
            ([batch_size,hidden_dim],[batch_size,hidden_dim])
        """
        with tf.variable_scope('encoder'):
            cell_fw = tf.contrib.rnn.LSTMCell(
                self.hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            cell_bw = tf.contrib.rnn.LSTMCell(
                self.hps.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
            (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
                cell_fw, cell_bw, encoder_inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)
            # the sequence length of the encoder_inputs varies depending on the
            # batch, which will make the second dimension of the
            # encoder_outputs different in different batches

            # concatenate the forwards and backwards states
            encoder_outputs = tf.concat(axis=2, values=encoder_outputs)
            # encoder_outputs: [batch_size * beam_size, max_time, output_size*2]
            # fw_st & bw_st: [batch_size * beam_size, num_hidden]
        return encoder_outputs, fw_st, bw_st

    def _reduce_states(self, fw_st, bw_st):
        """Add to the graph a linear layer to reduce the encoder's final FW and
        BW state into a single initial state for the decoder. This is needed
        because the encoder is bidirectional but the decoder is not.

        Args:
          fw_st: LSTMStateTuple with hidden_dim units.
          bw_st: LSTMStateTuple with hidden_dim units.

        Returns:
          state: LSTMStateTuple with hidden_dim units.
        """
        hidden_dim = self.hps.hidden_dim
        with tf.variable_scope('reduce_final_st'):

            # Define weights and biases to reduce the cell and reduce the state
            w_reduce_c = tf.get_variable(
                'w_reduce_c', [hidden_dim * 2, hidden_dim],
                dtype=tf.float32, initializer=self.trunc_norm_init)
            w_reduce_h = tf.get_variable(
                'w_reduce_h', [hidden_dim * 2, hidden_dim],
                dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_c = tf.get_variable(
                'bias_reduce_c', [hidden_dim],
                dtype=tf.float32, initializer=self.trunc_norm_init)
            bias_reduce_h = tf.get_variable(
                'bias_reduce_h', [hidden_dim],
                dtype=tf.float32, initializer=self.trunc_norm_init)

            # Apply linear layer
            # Concatenation of fw and bw cell
            old_c = tf.concat(axis=1, values=[fw_st.c, bw_st.c])
            # Concatenation of fw and bw state
            old_h = tf.concat(axis=1, values=[fw_st.h, bw_st.h])
            # [batch_size * beam_size, hidden_dim]
            new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)  # Get new cell from old cell
            new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)  # Get new state from old state
            return tf.contrib.rnn.LSTMStateTuple(new_c, new_h)  # Return new cell and state

    def _calc_final_dist(self, p_gens, vocab_dists, attn_dists):
        # this is the core function
        """Calculate the final distribution, for the pointer-generator model

        Args:
          vocab_dists: The vocabulary distributions. List length max_dec_steps
          of (batch_size, vsize) arrays. The words are in the order they appear
          in the vocabulary file.
          attn_dists: The attention distributions. List length max_dec_steps of
          (batch_size, attn_len) arrays

        Returns:
          final_dists: The final distributions. List length max-dec_steps of
          (batch_size, extended_vsize) arrays.
        """
        batch_size = tf.shape(vocab_dists[0])[0]
        with tf.variable_scope('final_distribution'):
            # Multiply vocab dists by p_gen and attention dists by (1-p_gen)
            # these three variable is confusing: vocab_dists, p_gens and
            # attn_dists
            vocab_dists = [
                p_gen * dist
                for (p_gen, dist) in zip(p_gens, vocab_dists)]
            # vocab_dists [max_dec_steps * (batch_size, vsize)]
            attn_dists = [
                (1 - p_gen) * dist
                for (p_gen, dist) in zip(p_gens, attn_dists)]
            # vocab_dists [max_dec_steps * (batch_size, attn_len)]

            # Concatenate some zeros to each vocabulary dist, to hold the
            # probabilities for in-article OOV words
            # the maximum (over the batch) size of the extended vocabulary
            extended_vsize = self._vocab.size() + self.max_art_oovs
            extra_zeros = tf.zeros((batch_size, self.max_art_oovs))
            vocab_dists_extended = [
                tf.concat(axis=1, values=[dist, extra_zeros])
                for dist in vocab_dists]
            # list length max_dec_steps of shape (batch_size, extended_vsize)

            # Project the values in the attention distributions onto the
            # appropriate entries in the final distributions
            # This means that if a_i = 0.1 and the ith encoder word is w, and w
            # has index 500 in the vocabulary, then we add 0.1 onto the 500th
            # entry of the final distribution
            # This is done for each decoder timestep.
            # This is fiddly; we use tf.scatter_nd to do the projection
            batch_nums = tf.range(0, limit=batch_size)
            # shape (batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)
            # shape (batch_size, 1)
            attn_len = tf.shape(self.enc_batch_extend_vocab)[1]
            # number of states we attend over
            # this is too tedious
            # shape (batch_size, attn_len)
            batch_nums = tf.tile(batch_nums, [1, attn_len])
            indices = tf.stack((batch_nums, self.enc_batch_extend_vocab), axis=2)
            # shape (batch_size, enc_t, 2)
            # what is this enc_batch_extend_vocab?
            shape = [batch_size, extended_vsize]
            attn_dists_projected = [
                tf.scatter_nd(indices, copy_dist, shape)
                for copy_dist in attn_dists]
            # this causes the error in the rollout!
            # a detailed article should be written about this
            # list length max_dec_steps (batch_size, extended_vsize)

            # Add the vocab distributions and the copy distributions together to
            # get the final distributions
            # final_dists is a list length max_dec_steps; each entry is a tensor
            # shape (batch_size, extended_vsize) giving the final distribution
            # for that decoder timestep
            # Note that for decoder timesteps and examples corresponding to a
            # [PAD] token, this is junk - ignore.
            final_dists = [
                vocab_dist + copy_dist for (vocab_dist, copy_dist) in zip(
                    vocab_dists_extended, attn_dists_projected)]

            # OOV part of vocab is max_art_oov long. Not all the sequences in a
            # batch will have max_art_oov tokens.  That will cause some entries
            # to be 0 in the distribution, which will result in NaN when
            # calulating log_dists Add a very small number to prevent that.

            def add_epsilon(dist, epsilon=sys.float_info.epsilon):
                epsilon_mask = tf.ones_like(dist) * epsilon
                return dist + epsilon_mask

            final_dists = [add_epsilon(dist) for dist in final_dists]

            return final_dists

    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self.hps

        with tf.name_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(
                -hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123)
            self.trunc_norm_init = tf.truncated_normal_initializer(stddev=hps.trunc_norm_init_std)

            k_samples_ls = tf.unstack(self.k_samples, axis=0)
            k_sample_targets_ls = tf.unstack(self.k_sample_targets, axis=0)
            k_sample_targets_mask_ls = tf.unstack(self.k_sample_targets_mask, axis=0)
            k_rewards_ls = tf.unstack(self.k_rewards, axis=0)

            with tf.variable_scope('embeddings'):
                self.embeddings = tf.get_variable(
                    'embeddings', [self._vocab.size(), hps.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
                self.saver = tf.train.Saver({"embeddings": self.embeddings})
                emb_enc_inputs = tf.nn.embedding_lookup(self.embeddings, self.enc_batch)
                # for gen training(mode is pretrain_gen) and
                # beam searching(mode is decode or train_gan)
                emb_dec_inputs = [tf.nn.embedding_lookup(self.embeddings, x)
                                  for x in tf.unstack(self._dec_batch, axis=1)]
                # for evaluation gan(when mode is train_gan)
                emb_eval_dec_inputs = [tf.nn.embedding_lookup(self.embeddings, x)
                                       for x in tf.unstack(self._eval_dec_batch, axis=1)]

                k_emb_samples_ls = [
                    [tf.nn.embedding_lookup(self.embeddings, x)
                     for x in tf.unstack(samples, axis=1)]
                    for samples in k_samples_ls
                ]

            # Add the encoder.
            enc_outputs, fw_st, bw_st = self._add_encoder(emb_enc_inputs, self.enc_lens)
            self.enc_states = enc_outputs
            self.dec_in_state = self._reduce_states(fw_st, bw_st)

            with tf.variable_scope('decoder') as decoder_scope:
                self.attn_dists, self.p_gens, self.coverage, \
                    final_dists, self._dec_out_state = self._add_decoder(emb_dec_inputs, self.dec_in_state)
                decoder_scope.reuse_variables()

                eval_attn_dists, _, eval_coverage, eval_final_dists, _ \
                    = self._add_decoder(emb_eval_dec_inputs, self.dec_in_state)

                k_sample_final_dists_ls = []
                for emb_samples in k_emb_samples_ls:
                    _, _, _, sample_final_dists, _ = self._add_decoder(emb_samples, self.dec_in_state)
                    k_sample_final_dists_ls.append(sample_final_dists)

            def get_loss(final_dists, target_batch, padding_mask, rewards=None):
                batch_nums = tf.range(0, limit=tf.shape(target_batch)[0])

                loss_per_step = []
                for dec_step, dist in enumerate(final_dists):
                    targets = target_batch[:, dec_step]
                    indices = tf.stack((batch_nums, targets), axis=1)
                    gold_probs = tf.gather_nd(dist, indices)
                    losses = -tf.log(gold_probs) * padding_mask[:, dec_step]
                    loss_per_step.append(losses * rewards[:, dec_step] if rewards is not None else losses)
                return loss_per_step

            # Calculate the loss
            with tf.variable_scope('generator_loss'):

                # for training of generator
                loss_per_step = get_loss(final_dists, self.target_batch, self.dec_padding_mask)
                eval_loss_per_step = get_loss(eval_final_dists, self.target_batch, self.dec_padding_mask)
                # Apply padding_mask mask and get loss
                self._loss = _avg(loss_per_step, self.dec_padding_mask)
                self._eval_loss = _avg(eval_loss_per_step, self.dec_padding_mask)

                # for training of GAN
                # Calculate coverage loss from the attention distributions
                if hps.coverage:
                    with tf.variable_scope('coverage_loss'):
                        self._coverage_loss = _coverage_loss(
                            self.attn_dists, self.dec_padding_mask)
                    self._total_loss = \
                        self._loss + hps.cov_loss_wt * self._coverage_loss

            with tf.variable_scope('gan_loss'):
                k_gan_losses = []
                for k in range(len(k_sample_targets_ls)):
                    gan_loss_per_step = get_loss(
                        k_sample_final_dists_ls[k], k_sample_targets_ls[k],
                        k_sample_targets_mask_ls[k], k_rewards_ls[k])
                    k_gan_losses.append(_avg(gan_loss_per_step, k_sample_targets_mask_ls[k]))

                self.gan_loss = tf.reduce_mean(tf.stack(k_gan_losses))

        # We run decode beam search mode one decoder step at a time
        # log_dists is a singleton list containing shape (batch_size,
        # extended_vsize)
        if len(emb_dec_inputs) == 1:
            assert len(final_dists) == 1
            self.final_dists = final_dists[0]
            topk_probs, self._topk_ids = tf.nn.top_k(
                self.final_dists, hps.beam_size * 2)
            self._topk_log_probs = tf.log(topk_probs)

            # for the monte carlo searching
            self._ran_id = tf.multinomial(tf.log(self.final_dists), 1)

        # for the loss
        loss_to_minimize = self._total_loss if self.hps.coverage else self._loss
        trainable_variables = tf.trainable_variables()
        gradients = tf.gradients(
            loss_to_minimize, trainable_variables,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

        # Clip the gradients
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(
                gradients, self.hps.gen_max_gradient)

        # Apply adagrad optimizer
        optimizer = tf.train.AdamOptimizer(self.hps.gen_lr)
        with tf.device("/gpu:0"):
            self._train_op = optimizer.apply_gradients(
                zip(grads, trainable_variables),
                global_step=self.global_step)

        # for the loss
        g_opt = self.g_optimizer(self.hps.gen_lr)
        trainable_variables = tf.trainable_variables()
        gradients = tf.gradients(self.gan_loss, trainable_variables,
                                 aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        self.g_grad, _ = tf.clip_by_global_norm(gradients, self.hps.gen_max_gradient)
        with tf.device("/gpu:0"):
            self.g_updates = g_opt.apply_gradients(zip(self.g_grad, trainable_variables), global_step=self.global_step)

        return decoder_scope

    def _add_decoder(self, emb_dec_inputs, dec_in_state, is_sequence=False):
        """
        input:
            emb_dec_inputs, the input of the cell
        to get:
            output log distribution
            new state
        """
        vsize = self._vocab.size()  # size of the vocabulary
        # Add the decoder.
        cell = tf.contrib.rnn.LSTMCell(
            self.hps.hidden_dim,
            state_is_tuple=True,
            initializer=self.rand_unif_init)

        # In decode mode, we run attention_decoder one step at a time and so
        # need to pass in the previous step's coverage vector each time
        # a placeholder, why not a variable?
        prev_coverage = self.prev_coverage if self.hps.coverage and self.hps.mode in ["train_gan", "decode"] else None
        # coverage is for decoding in beam_search and gan training
        is_sequence = self.hps.mode == ["pretrain_gen"] or is_sequence

        outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(
            emb_dec_inputs, dec_in_state, self.enc_states, self.enc_padding_mask, cell,
            initial_state_attention=(len(emb_dec_inputs) > 1),
            use_coverage=self.hps.coverage, prev_coverage=prev_coverage)

        # Add the output projection to obtain the vocabulary distribution
        with tf.variable_scope('output_projection'):
            w = tf.get_variable(
                'w', [self.hps.hidden_dim, vsize],
                dtype=tf.float32, initializer=self.trunc_norm_init)
            w_t = tf.transpose(w)  # NOQA
            v = tf.get_variable('v', [vsize], dtype=tf.float32, initializer=self.trunc_norm_init)
            vocab_scores = []
            # vocab_scores is the vocabulary distribution before applying
            # softmax. Each entry on the list corresponds to one decoder
            # step
            for i, output in enumerate(outputs):
                vocab_scores.append(tf.nn.xw_plus_b(output, w, v))
                # apply the linear layer

            # The vocabulary distributions. List length max_dec_steps of
            # (batch_size, vsize) arrays. The words are in the order they
            # appear in the vocabulary file.
            vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]
            # if not FLAGS.pointer_gen:  # calculate loss from log_dists
            #     self.vocab_scores = vocab_scores
            # is the oov included

        # For pointer-generator model, calc final distribution from copy
        # distribution and vocabulary distribution, then take log
        final_dists = self._calc_final_dist(p_gens, vocab_dists, attn_dists)
        return attn_dists, p_gens, coverage, final_dists, out_state

    def build_graph(self):
        """Add the placeholders, model, global step, train_op and summaries to
        the graph"""
        t0 = time.time()
        self._add_placeholders()
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        with tf.device("/gpu:0"):
            decoder_scope = self._add_seq2seq()
        self.least_val_loss = tf.Variable(1000.0, name='least_val_loss', trainable=False)
        t1 = time.time()
        print(colored('Time to build graph: %s seconds' % (t1 - t0), "yellow"))
        return decoder_scope

    def run_one_batch(self, sess, batch, update=True, gan_eval=False):
        """Runs one training iteration. Returns a dictionary containing train
        op, summaries, loss, global_step and (optionally) coverage loss."""
        if gan_eval:
            update = False

        feed_dict = self._make_feed_dict(batch, gan_eval=gan_eval)

        to_return = {
            'global_step': self.global_step,
        }
        if gan_eval:
            to_return['loss'] = self._eval_loss
        else:
            to_return['loss'] = self._loss
        if update:
            # if update is False it is for the generator evaluation
            to_return['train_op'] = self._train_op
        if self.hps.coverage:
            to_return['coverage_loss'] = self._coverage_loss
        return sess.run(to_return, feed_dict)

    def run_gan_batch(
        self, sess, batch, samples, sample_targets,
        sample_padding_mask, rewards, update=True, gan_eval=False
    ):
        feed_dict = self._make_feed_dict(batch, gan_eval=gan_eval, gan=True)

        # this can be combined with evaluation method
        feed_dict.update({
            # for the decoder
            self.k_samples: samples,
            self.k_sample_targets: sample_targets,
            self.k_sample_targets_mask: sample_padding_mask,
            self.k_rewards: rewards,
        })

        to_return = {
            'global_step': self.global_step,
            'loss': self.gan_loss,
        }
        if update:
            to_return['updates'] = self.g_updates
        return sess.run(to_return, feed_dict)

    def run_encoder(self, sess, batch):
        """For beam search decoding. Run the encoder on the batch and return the
        encoder states and decoder initial state.

        Args:
          sess: Tensorflow session.
          batch: Batch object that is the same example repeated across the batch
          (for beam search)

        Returns:
          enc_states: The encoder states. A tensor of shape [batch_size,
          <=max_enc_steps, 2*hidden_dim].
          dec_in_state: A LSTMStateTuple of shape
          ([batch_size, hidden_dim],[batch_size, hidden_dim])
        """
        feed_dict = self._make_feed_dict(batch, just_enc=True)
        # feed the batch into the placeholders
        (enc_states, dec_in_state) = sess.run(
            [
                self.enc_states,
                self.dec_in_state,
            ],
            feed_dict
        )  # run the encoder
        # enc_states: [batch_size * beam_size, <=max_enc_steps, 2*hidden_dim]
        # dec_in_state: [batch_size * beam_size, ]

        # dec_in_state is LSTMStateTuple shape
        # ([batch_size,hidden_dim],[batch_size,hidden_dim])
        # Given that the batch is a single example repeated, dec_in_state is
        # identical across the batch so we just take the top row.
        # dec_in_state = [tf.contrib.rnn.LSTMStateTuple(
        #     dec_in_state.c[i], dec_in_state.h[i] for i in xrange(len(dec_in_state.h))]
        #     # TODO: should this be changed to shape?
        return enc_states, dec_in_state

    def decode_onestep(self, emb_dec_inputs, dec_in_state):
        """
        function: decode onestep for rollout
        inputs:
            the embedded input
        """
        # attn_dists, p_gens, coverage, vocab_scores, log_probs, new_states
        _, _, _, final_dists, new_states = self._add_decoder(emb_dec_inputs, dec_in_state, is_sequence=True)
        # how can it be fed by a [batch_size * 1 * emb_dim] while decoding?
        # final_dists_sliced = tf.slice(final_dists[0], [0, 0], [-1, self._vocab.size()])
        final_dists = final_dists[0]
        final_dists += tf.ones_like(final_dists) * sys.float_info.epsilon
        output_id = tf.squeeze(tf.cast(tf.reshape(tf.multinomial(tf.log(final_dists), 1), [self.hps.batch_size]), tf.int32))
        # next_input = tf.nn.embedding_lookup(self.embeddings, next_token)  # batch x emb_dim
        return output_id, new_states

    def run_decode_onestep(self, sess, enc_batch_extend_vocab, max_art_oovs,
                           latest_tokens, enc_states, enc_padding_mask,
                           dec_init_states, prev_coverage, method="bs"):
        """For beam search decoding. Run the decoder for one step.

        Args:
          sess: Tensorflow session.
          enc_batch_extend_vocab: the encode batch with extended vocabulary
          max_art_oovs: the max article out of vocabulary
          latest_tokens: Tokens to be fed as input into the decoder for this
          timestep
          enc_states: The encoder states.
          dec_init_states: List of beam_size LSTMStateTuples; the decoder states
          from the previous timestep
          prev_coverage: List of np arrays. The coverage vectors from the
          previous timestep. List of None if not using coverage.
          method: can be bs standing for beam search and mc representing monte carlo

        Returns:
          ids: top 2k ids. shape [beam_size, 2*beam_size]
          probs: top 2k log probabilities. shape [beam_size, 2*beam_size]
          new_states: new states of the decoder. a list length beam_size
          containing
            LSTMStateTuples each of shape ([hidden_dim,],[hidden_dim,])
          attn_dists: List length beam_size containing lists length attn_length.
          p_gens: Generation probabilities for this step. A list length
          beam_size. List of None if in baseline mode.
          new_coverage: Coverage vectors for this step. A list of arrays. List
          of None if coverage is not turned on.
        """

        # Turn dec_init_states (a list of LSTMStateTuples) into a single
        # LSTMStateTuple for the batch
        cells = [np.expand_dims(state.c, axis=0) for state in dec_init_states]
        hiddens = [np.expand_dims(state.h, axis=0) for state in dec_init_states]
        new_c = np.concatenate(cells, axis=0)  # shape [batch_size,hidden_dim]
        new_h = np.concatenate(hiddens, axis=0)  # shape [batch_size,hidden_dim]
        new_dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

        batch_size = enc_states.shape[0]

        feed = {
            self.enc_states: enc_states,
            self.enc_padding_mask: enc_padding_mask,
            self.dec_in_state: new_dec_in_state,
            self._dec_batch: latest_tokens,
            self.enc_batch_extend_vocab: enc_batch_extend_vocab,
            self.max_art_oovs: max_art_oovs,
        }

        to_return = {
          "states": self._dec_out_state,
          "attn_dists": self.attn_dists,
          "final_dists": self.final_dists,
          "p_gens": self.p_gens,
        }

        if method == "bs":
            to_return["ids"] = self._topk_ids
            to_return["probs"] = self._topk_log_probs
        elif method == "mc":
            to_return["ran_id"] = self._ran_id

        if self.hps.coverage:
            feed[self.prev_coverage] = prev_coverage
            to_return['coverage'] = self.coverage

        results = sess.run(to_return, feed_dict=feed)  # run the decoder step

        # Convert results['states'] (a single LSTMStateTuple) into a list of
        # LSTMStateTuple -- one for each hypothesis
        new_states = [
            tf.contrib.rnn.LSTMStateTuple(
                results['states'].c[i, :], results['states'].h[i, :])
            for i in xrange(batch_size)]

        if method == "bs":
            sample_ids = results["ids"]
            probs = results["probs"]

            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['attn_dists']) == 1
            attn_dists = results['attn_dists'][0].tolist()

            # Convert singleton list containing a tensor to a list of k arrays
            assert len(results['p_gens']) == 1
            p_gens = results['p_gens'][0].tolist()
        else:
            sample_ids = results["ran_id"]
            probs = None

            attn_dists = None
            p_gens = None

        # Convert the coverage tensor to a list length k containing the coverage
        # vector for each hypothesis
        if FLAGS.coverage:
            new_coverage = results['coverage'].tolist()
            assert len(new_coverage) == batch_size
        else:
            new_coverage = [None for _ in xrange(enc_states.shape[1])]

        return sample_ids, probs, new_states, attn_dists, p_gens, new_coverage

    def g_optimizer(self, *args, **kwargs):
        return tf.train.AdamOptimizer(*args, **kwargs)


def _mask_and_avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and
      0s.

    Returns:
      a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    # shape (batch_size); normalized value for each batch member
    values_per_ex = tf.reduce_sum(tf.stack(values_per_step, 1), 1)/dec_lens
    return tf.reduce_mean(values_per_ex)  # overall average


def _avg(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and
      0s.

    Returns:
      a scalar
    """

    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # shape batch_size. float32
    # values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    # shape (batch_size); normalized value for each batch member
    values_per_ex = tf.reduce_sum(tf.stack(values, 1), 1)/dec_lens
    return tf.reduce_mean(values_per_ex)  # overall average


def _mask(values, padding_mask):
    """Applies mask to values then returns overall average (a scalar)

    Args:
      values: a list length max_dec_steps containing arrays shape (batch_size).
      padding_mask: tensor shape (batch_size, max_dec_steps) containing 1s and
      0s.

    Returns:
      a scalar
    """

    values_per_step = [v * padding_mask[:, dec_step] for dec_step, v in enumerate(values)]
    # shape (batch_size); normalized value for each batch member
    values_per_ex = sum(values_per_step)
    return tf.reduce_sum(values_per_ex)  # overall loss


def _coverage_loss(attn_dists, padding_mask):
    """Calculates the coverage loss from the attention distributions.

    Args:
      attn_dists: The attention distributions for each decoder timestep. A list
      length max_dec_steps containing shape (batch_size, attn_length)
      padding_mask: shape (batch_size, max_dec_steps).

    Returns:
      coverage_loss: scalar
    """
    coverage = tf.zeros_like(
        attn_dists[0])
    # shape (batch_size, attn_length). Initial coverage is zero.
    # Coverage loss per decoder timestep. Will be list length max_dec_steps
    # containing shape (batch_size).
    covlosses = []
    for a in attn_dists:
        # calculate the coverage loss for this step
        covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])
        covlosses.append(covloss)
        coverage += a  # update the coverage vector
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    return coverage_loss
