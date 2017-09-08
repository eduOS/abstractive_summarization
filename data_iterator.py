import cPickle as pkl
import gzip
import numpy


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class disThreeTextIterator:

    def __init__(
        self, positive_data, negative_data, source_data,
        vocab, vocab_size, batch=1, maxlen=80, dismaxlen=15
    ):
        self.positive = fopen(positive_data, 'r')
        self.negative = fopen(negative_data, 'r')
        self.source = fopen(source_data, 'r')
        self.vocab = vocab
        self.vocab_size = vocab_size

        self.batch_size = batch
        assert self.batch_size % 2 == 0
        self.maxlen = maxlen
        self.dismaxlen = dismaxlen
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.positive.seek(0)
        self.negative.seek(0)
        self.source.seek(0)

    def next(self):  # NOQA
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        positive = []
        negative = []
        source = []
        x = []
        xs = []
        y = []

        try:
            while True:
                ss = self.positive.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.vocab.word2id(w) for w in ss]
                if self.vocab_size > 0:
                    ss = [w if w < self.vocab_size else 0 for w in ss]

                tt = self.negative.readline()
                if tt == "":
                    # raise IOError
                    continue
                tt = tt.strip().split()
                tt = [self.vocab.word2id(w) for w in tt]
                if self.vocab_size > 0:
                    tt = [w if w < self.vocab_size else 0 for w in tt]

                ll = self.source.readline()
                if ll == "":
                    raise IOError
                ll = ll.strip().split()
                ll = [self.vocab.word2id(w) for w in ll]
                if self.vocab_size > 0:
                    ll = [w if w < self.vocab_size else 0 for w in ll]

                if (
                    len(ss) > self.dismaxlen or
                    len(tt) > self.dismaxlen or
                    len(ll) > self.maxlen
                ):
                    continue

                positive.append(ss)
                negative.append(tt)
                source.append(ll)

                x = positive + negative

                positive_labels = [[0, 1] for _ in positive]
                negative_labels = [[1, 0] for _ in negative]
                y = positive_labels + negative_labels

                xs = source + source

                shuffle_indices = numpy.random.permutation(numpy.arange(len(x)))
                x_np = numpy.array(x)
                y_np = numpy.array(y)
                xs_np = numpy.array(xs)

                x_np_shuffled = x_np[shuffle_indices]
                y_np_shuffled = y_np[shuffle_indices]
                xs_np_shuffled = xs_np[shuffle_indices]

                x_shuffled = x_np_shuffled.tolist()
                y_shuffled = y_np_shuffled.tolist()
                xs_shuffled = xs_np_shuffled.tolist()

                if (
                    len(x_shuffled) >= self.batch_size and
                    len(y_shuffled) >= self.batch_size and
                    len(xs_shuffled) >= self.batch_size
                ):
                    break
        except IOError:
            self.end_of_data = True

        if len(positive) <= 0 or len(negative) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return x_shuffled, y_shuffled, xs_shuffled


class disTextIterator:

    def __init__(
        self,
        positive_data,
        negative_data,
        dis_dict,
        batch=1,
        maxlen=30,
        dismaxlen=-1
    ):
        self.positive = fopen(positive_data, 'r')
        self.negative = fopen(negative_data, 'r')
        # what is the positive and negative data?
        with open(dis_dict) as f:
            self.dis_dict = pkl.load(f)

        self.batch_size = batch
        assert self.batch_size % 2 == 0, \
            'the batch size of disTextIterator is not an even number'

        self.maxlen = maxlen
        self.dismaxlen = dismaxlen
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.positive.seek(0)
        self.negative.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        positive = []
        negative = []
        x = []
        y = []
        try:
            while True:
                ss = self.positive.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.dis_dict[w] if w in self.dis_dict else 1 for w in ss]
                if self.dismaxlen > 0:
                    ss = [w if w < self.dismaxlen else 1 for w in ss]

                tt = self.negative.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.dis_dict[w] if w in self.dis_dict else 1 for w in tt]
                if self.dismaxlen > 0:
                    tt = [w if w < self.dismaxlen else 1 for w in tt]

                if len(ss) > self.maxlen or len(tt) > self.maxlen:
                    continue

                positive.append(ss)
                negative.append(tt)
                x = positive + negative
                positive_labels = [[0, 1] for _ in positive]
                negative_labels = [[1, 0] for _ in negative]
                y = positive_labels + negative_labels
                shuffle_indices = numpy.random.permutation(numpy.arange(len(x)))
                x_np = numpy.array(x)
                y_np = numpy.array(y)
                x_np_shuffled = x_np[shuffle_indices]
                y_np_shuffled = y_np[shuffle_indices]

                x_shuffled = x_np_shuffled.tolist()
                y_shuffled = y_np_shuffled.tolist()

                if len(x_shuffled) >= self.batch_size and len(
                        y_shuffled) >= self.batch_size:
                    break

        except IOError:
            self.end_of_data = True

        if len(positive) <= 0 or len(negative) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return x_shuffled, y_shuffled


class genTextIterator:

    def __init__(
        self,
        train_data,
        source_dict,
        batch_size=1,
        maxlen=30,
        dismaxlen=-1
    ):
        self.source = fopen(train_data, 'r')

        with open(source_dict, 'rb') as f:
            self.source_dict = pkl.load(f)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.dismaxlen = dismaxlen
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        try:
            while True:
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [
                    self.source_dict[w]
                    if w in self.source_dict else 1 for w in ss]
                if self.dismaxlen > 0:
                    ss = [w if w < self.dismaxlen else 1 for w in ss]

                if len(ss) > self.maxlen:
                    continue

                source.append(ss)

                if len(source) >= self.batch_size:
                    break
        except:
            self.end_of_data = True

        if len(source) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source


class TextIterator:
    """Simple Bitext iterator."""

    def __init__(
        self, source, target, vocab,
        vocab_size, batch_size=128, max_len_s=50, max_leng=15,
    ):
        self.source = fopen(source, 'r')
        self.target = fopen(target, 'r')
        self.batch_size = batch_size
        self.max_len_s = max_len_s
        self.max_leng = max_leng
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.target.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []

        try:

            # actual work here
            while True:

                # read from source file and map to word index
                ss = self.source.readline()
                if ss == "":
                    raise IOError
                ss = ss.strip().split()
                ss = [self.vocab.word2id(w) for w in ss]
                if self.vocab_size > 0:
                    ss = [w if w < self.vocab_size else 0 for w in ss]

                # read from source file and map to word index
                tt = self.target.readline()
                if tt == "":
                    raise IOError
                tt = tt.strip().split()
                tt = [self.vocab.word2id(w) for w in tt]
                if self.vocab_size > 0:
                    tt = [w if w < self.vocab_size else 0 for w in tt]

                if len(ss) > self.max_len_s and len(tt) > self.max_leng:
                    continue

                source.append(ss)
                target.append(tt)

                if len(source) >= self.batch_size or \
                        len(target) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True

        if len(source) <= 0 or len(target) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target
