import tensorflow as tf
import numpy
import time
import os

from data_iterator import disTextIterator
from data_iterator import genTextIterator
from data_iterator import TextIterator

PAD_TOKEN = '[PAD]'
# This has a vocab id, which is used to represent out-of-vocabulary words
UNKNOWN_TOKEN = '[UNK]'
# This has a vocab id, which is used at the start of every decoder input
# sequence
START_DECODING = '[START]'
# This has a vocab id, which is used at the end of untruncated target sequences
STOP_DECODING = '[STOP]'


def prepare_gan_dis_data(
    train_data_source,
    train_data_target,
    gan_dis_source_data,
    gan_dis_positive_data,
    num=None,
    reshuf=True
):

    source = open(train_data_source, 'r')
    sourceLists = source.readlines()

    if num is None or num > len(sourceLists):
        num = len(sourceLists)

    if reshuf:
        os.popen('python shuffle.py ' + train_data_source+' '+train_data_target)
        os.popen(
            'head -n ' +
            str(num) +
            ' ' +
            train_data_source +
            '.shuf' +
            ' >' +
            gan_dis_source_data)
        os.popen(
            'head -n ' +
            str(num) +
            ' ' +
            train_data_target +
            '.shuf' +
            ' >' +
            gan_dis_positive_data)
    else:
        os.popen(
            'head -n ' +
            str(num) +
            ' ' +
            train_data_source +
            '.shuf' +
            ' >' +
            gan_dis_source_data)
        os.popen(
            'head -n ' +
            str(num) +
            ' ' +
            train_data_target +
            '.shuf' +
            ' >' +
            gan_dis_positive_data)

    os.popen('rm '+train_data_source+'.shuf')
    os.popen('rm '+train_data_target+'.shuf')
    return num


def prepare_three_gan_dis_dev_data(
    gan_dis_positive_data,
    gan_dis_negative_data,
    gan_dis_source_data,
    dev_dis_positive_data,
    dev_dis_negative_data,
    dev_dis_source_data,
    num
):
    gan_dis = open(gan_dis_positive_data, 'r')
    disLists = gan_dis.readlines()

    if num is None or num > len(disLists):
        num = len(disLists)

    os.popen(
        'head -n ' +
        str(num) +
        ' ' +
        gan_dis_positive_data +
        ' >' +
        dev_dis_positive_data)
    os.popen(
        'head -n ' +
        str(num) +
        ' ' +
        gan_dis_negative_data +
        ' >' +
        dev_dis_negative_data)
    os.popen(
        'head -n ' +
        str(num) +
        ' ' +
        gan_dis_source_data +
        ' >' +
        dev_dis_source_data)

    return num


def prepare_gan_dis_dev_data(
    gan_dis_positive_data,
    gan_dis_negative_data,
    dev_dis_positive_data,
    dev_dis_negative_data,
    num
):

    gan_dis = open(gan_dis_positive_data, 'r')
    disLists = gan_dis.readlines()

    if num is None or num > len(disLists):
        num = len(disLists)

    os.popen(
        'head -n ' +
        str(num) +
        ' ' +
        gan_dis_positive_data +
        ' >' +
        dev_dis_positive_data)
    os.popen(
        'head -n ' +
        str(num) +
        ' ' +
        gan_dis_negative_data +
        ' >' +
        dev_dis_negative_data)

    return num


def print_string(indexs, vocab):
    sample_str = ''
    for index in indexs:
        if index > 0:
            word_str = vocab.id2word(index)
            sample_str = sample_str + word_str + ' '
    return sample_str


class FlushFile:
    """
    A wrapper for File, allowing users see result immediately.
    """

    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()


def _p(pp, name):
    return '%s_%s' % (pp, name)


def dis_train_iter(
    dis_positive_data,
    dis_negative_data,
    reshuffle,
    dictionary,
    n_words_trg,
    batch_size,
    maxlen
):
    iter = 0
    while True:
        if reshuffle:
            os.popen(
                'python shuffle.py ' + dis_positive_data +
                ' ' + dis_positive_data)
            os.popen('mv ' + dis_negative_data + '.shuf ' + dis_negative_data)
            os.popen('mv ' + dis_negative_data + '.shuf ' + dis_negative_data)
        disTrain = disTextIterator(
            dis_positive_data,
            dis_negative_data,
            dictionary,
            batch_size,
            maxlen,
            n_words_trg
        )
        iter += 1
        ExampleNum = 0
        for x, y in disTrain:
            ExampleNum += len(x)
            yield x, y, iter


def gen_train_iter(
    gen_file,
    reshuffle,
    dictionary,
    n_words,
    batch_size,
    maxlen
):
    iter = 0
    while True:
        if reshuffle:
            os.popen('python shuffle.py ' + gen_file)
            os.popen('mv ' + gen_file + '.shuf ' + gen_file)
        gen_train = genTextIterator(
            gen_file,
            dictionary,
            n_words_source=n_words,
            batch_size=batch_size,
            maxlen=maxlen
        )
        ExampleNum = 0
        EpochStart = time.time()
        for x in gen_train:
            if len(x) < batch_size:
                continue
            ExampleNum += len(x)
            yield x, iter
        TimeCost = time.time() - EpochStart
        iter += 1
        print('Seen ', ExampleNum, 'generator samples. Time cost is ', TimeCost)


def gen_force_train_iter(
    source_data,
    target_data,
    reshuffle,
    vocab,
    vocab_size,
    batch_size,
    max_len_s,
    max_leng,
):
    iter_num = 0
    while True:
        if reshuffle:
            os.popen('python shuffle.py ' + source_data + ' ' + target_data)
            os.popen('mv ' + source_data + '.shuf ' + source_data)
            os.popen('mv ' + target_data + '.shuf ' + target_data)
        gen_force_train = TextIterator(
            source_data,
            target_data,
            vocab,
            vocab_size,
            batch_size,
            max_len_s,
            max_leng)
        ExampleNum = 0
        EpochStart = time.time()
        for x, y in gen_force_train:
            if len(x) < batch_size and len(y) < batch_size:
                continue
            ExampleNum += len(x)
            yield x, y, iter_num
        TimeCost = time.time() - EpochStart
        iter_num += 1
        print('Seen', ExampleNum, 'generator samples. Time cost is ', TimeCost)


def prepare_data(seqs_x, seqs_y, max_len_s=None, max_leng=None,
                 vocab_size=30000, precision='float32'):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if max_len_s is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < max_len_s:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

    if max_leng is not None:
        new_seqs_y = []
        new_lengths_y = []
        for l_y, s_y in zip(lengths_y, seqs_y):
            if l_y < max_leng:
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int32')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype(precision)
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype(precision)
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


def dis_three_length_prepare(seqs_x, seqs_y, seqs_xs, maxlen=50, dismaxlen=15):
    n_samples = len(seqs_x)
    x = numpy.zeros((dismaxlen, n_samples)).astype('int32')
    y = numpy.zeros((2, n_samples)).astype('int32')
    xs = numpy.zeros((maxlen, n_samples)).astype('int32')

    for idx, [s_x, s_y, s_xs] in enumerate(zip(seqs_x, seqs_y, seqs_xs)):
        x[:len(s_x), idx] = s_x
        y[:len(s_y), idx] = s_y
        xs[:len(s_xs), idx] = s_xs
    return x, y, xs


def dis_length_prepare(seqs_x, seqs_y, maxlen=50):
    n_samples = len(seqs_x)
    x = numpy.zeros((maxlen, n_samples)).astype('int32')
    y = numpy.zeros((2, n_samples)).astype('int32')

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:len(s_x), idx] = s_x
        y[:len(s_y), idx] = s_y
    return x, y


def prepare_single_sentence(seqs_x, maxlen=50):
    n_samples = len(seqs_x)
    lens_x = [len(seq) for seq in seqs_x]
    maxlen_x = numpy.max(lens_x) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int32')
    for idx, s_x in enumerate(seqs_x):
        x[:len(s_x), idx] = s_x
    return x


def prepare_multiple_sentence(seqs_x, maxlen=50, precision='float32'):
    n_samples = len(seqs_x)
    lens_x = [len(seq) for seq in seqs_x]
    maxlen_x = numpy.max(lens_x) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype(precision)

    for idx, s_x in enumerate(seqs_x):
        x[:len(s_x), idx] = s_x
        x_mask[:len(s_x), idx] = 1.

    return x, x_mask


def prepare_sentence_to_maxlen(seqs_x, maxlen=50, precision='float32'):
    n_samples = len(seqs_x)
    x = numpy.zeros((maxlen, n_samples)).astype('int32')

    for idx, s_x in enumerate(seqs_x):
        x[:len(s_x), idx] = s_x
    return x


def deal_generated_y_sentence(seqs_y, vocab, precision='float32'):
    n_samples = len(seqs_y)
    lens_y = [len(seq) for seq in seqs_y]
    maxlen_y = numpy.max(lens_y)
    eosTag = '[STOP]'
    eosIndex = vocab.word2id(eosTag)

    y = numpy.zeros((maxlen_y, n_samples)).astype('int32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype(precision)

    for idy, s_y in enumerate(seqs_y):
        try:
            firstIndex = s_y.tolist().index(eosIndex)+1
        except ValueError:
            firstIndex = maxlen_y - 1

        y[:firstIndex, idy] = s_y[:firstIndex]
        y_mask[:firstIndex, idy] = 1.

    return y, y_mask


def ortho_weight(ndim, precision='float32'):
    W = numpy.random.randn(ndim, ndim)
    u, s, v = numpy.linalg.svd(W)
    return u.astype(precision)


def norm_weight(nin, nout=None, scale=0.01, ortho=True, precision='float32'):
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        W = scale * numpy.random.randn(nin, nout)
    return W.astype(precision)


def tableLookup(
        vocab_size,
        embedding_size,
        scope="tableLookup",
        init_device='/cpu:0',
        reuse_var=False,
        prefix='tablelookup'):

    if not scope:
        scope = tf.get_variable_scope()

    with tf.variable_scope(scope):
        if not reuse_var:
            with tf.device(init_device):
                embeddings_init = norm_weight(vocab_size, embedding_size)
                embeddings = tf.get_variable(
                    'embeddings', shape=[vocab_size, embedding_size],
                    initializer=tf.constant_initializer(embeddings_init))
        else:
            tf.get_variable_scope().reuse_variables()
            embeddings = tf.get_variable('embeddings')
    return embeddings


def FCLayer(
        state_below,
        input_size,
        output_size,
        is_3d=True,
        reuse_var=False,
        use_bias=True,
        activation=None,
        scope='ff',
        init_device='/cpu:0',
        prefix='ff',
        precision='float32'):
    # it is kind of like linear

    if not scope:
        scope = tf.get_variable_scope()

    with tf.variable_scope(scope):
        if not reuse_var:
            with tf.device(init_device):
                W_init = norm_weight(input_size, output_size)
                matrix = tf.get_variable(
                    'W', [input_size, output_size],
                    initializer=tf.constant_initializer(W_init),
                    trainable=True)
                if use_bias:
                    bias_init = numpy.zeros((output_size,)).astype(precision)
                    bias = tf.get_variable(
                        'b', output_size, initializer=tf.constant_initializer(
                            bias_init),
                        trainable=True)
        else:
            tf.get_variable_scope().reuse_variables()
            matrix = tf.get_variable('W')
            if use_bias:
                bias = tf.get_variable('b')

        inputShape = tf.shape(state_below)
        if is_3d:
            state_below = tf.reshape(state_below, [-1, inputShape[2]])
            output = tf.matmul(state_below, matrix)
            output = tf.reshape(output, [-1, inputShape[1], output_size])
        else:
            output = tf.matmul(state_below, matrix)
        if use_bias:
            output = tf.add(output, bias)
        if activation is not None:
            output = activation(output)
    return output


def average_clip_gradient(tower_grads, clip_c):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
                # Note that each grad_and_vars looks like the following:
                #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
            # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        #  across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    if clip_c > 0:
        grad, value = zip(*average_grads)
        grad, global_norm = tf.clip_by_global_norm(grad, clip_c)
        average_grads = zip(grad, value)

    # self.average_grads = average_grads

    return average_grads


def average_clip_gradient_by_value(tower_grads, clip_min, clip_max):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
            # Average over the 'tower' dimension.
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        # Keep in mind that the Variables are redundant because they are shared
        #  across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    if clip_max > 0:
        grad, value = zip(*average_grads)
        grad = [tf.clip_by_value(x, clip_min, clip_max) for x in grad]
        average_grads = zip(grad, value)

    return average_grads


class Vocab(object):
    """Vocabulary class for mapping between words and ids (integers)"""

    def __init__(self, vocab_file, max_size=200000):
        """Creates a vocab of up to max_size words, reading from the vocab_file.
        If max_size is 0, reads the entire vocab file.

        Args:
          vocab_file: path to the vocab file, which is assumed to contain
          "<word> <frequency>" on each line, sorted with most frequent word
          first. This code doesn't actually use the frequencies, though.
          max_size: integer. The maximum size of the resulting Vocabulary."""
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0

        for w in [
            UNKNOWN_TOKEN, START_DECODING, STOP_DECODING
        ]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1

        # Read the vocab file and add words up to max_size
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    print(
                        'Warning: incorrectly formatted line in vocabulary file:\
                        %s\n' % line)
                    continue
                w = pieces[0]
                # should I add the end of sentence?
                if w in [
                    UNKNOWN_TOKEN, START_DECODING, STOP_DECODING
                ]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t\
                        be in the vocab file, but %s is' % w)
                if w in self._word_to_id:
                    raise Exception(
                        'Duplicated word in vocabulary file: %s' % w)
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if max_size != 0 and self._count >= max_size:
                    print(
                        "max_size of vocab was specified as %i; we now have %i\
                        words. Stopping reading." %
                        (max_size, self._count))
                    break

        print(
            "Finished constructing vocabulary of %i total words. Last word\
            added: %s" % (
                self._count, self._id_to_word[self._count - 1]))

    def word2id(self, word):
        """Returns the id (integer) of a word (string). Returns [UNK] id if word
        is OOV."""
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        """Returns the word (string) corresponding to an id (integer)."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def size(self):
        """Returns the total size of the vocabulary"""
        return self._count


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
      scope: VariableScope for the created subgraph; defaults to "Linear".

    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("`args` must be specified")
    if not isinstance(args, (list, tuple)):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError(
                "Linear is expecting 2D arguments: %s" %
                str(shapes))
        if not shape[1]:
            raise ValueError(
                "Linear expects shape[1] of arguments: %s" %
                str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            "Bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term
