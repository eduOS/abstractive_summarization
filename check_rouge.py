# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from __future__ import absolute_import
from __future__ import division

import glob
from cntk.tokenizer import text2charlist
import numpy as np
from codecs import open
from collections import Counter
from pathlib import Path
from cntk.constants.stopwords import default as default_stopwords
import os


class BasicElement():

    def __init__(self, head, modifier, relation):
        self.head = head
        self.modifier = modifier
        self.relation = relation

    def equals(self, other, option="HMR"):
        equal = True
        for c in option:
            c = c.upper()
            if c == "H" and self.head != other.head:
                equal = False
            elif c == "M" and self.modifier != other.modifier:
                equal = False
            elif c == "R" and self.relation != other.relation:
                equal = False
        return equal

    def as_key(self, option="HMR"):
        els = []
        for c in option:
            c = c.upper()
            if c == "H":
                els.append(self.head)
            elif c == "M":
                els.append(self.modifier)
            elif c == "R":
                els.append(self.relation)
        return "|".join(els)

    def __repr__(self):
        return "<BasicElement: {}-[{}]->{}>".format(
                    self.head, self.relation, self.modifier)


class BaseLang(object):
    _PARSER = None

    def __init__(self, lang):
        self.lang = lang
        self._stopwords = []
        self._stemming = {}

    def load_parser(self):
        if self._PARSER is None:
            import spacy
            self._PARSER = spacy.load(self.lang)
        return self._PARSER

    def tokenize(self, text):
        raise Exception("Have to implement tokenize in subclass")

    def tokenized_str(self, text):
        return " ".join(self.tokenize(text))

    def parse_to_be(self, text):
        from spacy.symbols import VERB, ADJ
        doc = self.load_parser()(text)
        bes = []
        for chunk in doc.noun_chunks:
            # chunk level dependencies
            if chunk.root.head.pos in [VERB, ADJ]:
                be = BasicElement(chunk.root.text, chunk.root.head.lemma_,
                                  chunk.root.dep_,)
                bes.append(be)

                # in-chunk level dependencies
                for c in chunk.root.children:
                    if c.pos in [VERB, ADJ]:
                        be = BasicElement(chunk.root.text, c.lemma_,
                                          c.dep_)
                        bes.append(be)
        return bes

    def is_stop_word(self, word):
        if len(self._stopwords) == 0:
            self.load_stopwords()
        return word in self._stopwords

    def stemming(self, word, min_length=-1):
        if len(self._stemming) == 0:
            self.load_stemming_dict()

        _word = word
        if min_length > 0 and len(_word) < min_length:
            return _word
        elif _word in self._stemming:
            return self._stemming[_word]
        else:
            return _word
        return _word

    def load_stopwords(self):
        self._stopwords = default_stopwords

    def load_stemming_dict(self):
        p = Path(os.path.dirname(__file__))
        p = p.joinpath("data", self.lang, "stemming.txt")
        if p.is_file():
            with p.open(encoding="utf-8") as f:
                lines = f.readlines()
                lines = [ln.strip() for ln in lines]
                pairs = [ln.split(" ", 1) for ln in lines if ln]
            self._stemming = dict(pairs)


class RougeCalculator():

    def __init__(self,
                 stopwords=True, stemming=False,
                 word_limit=-1, length_limit=-1, tokenizer=None):
        self.stemming = stemming
        self.stopwords = stopwords
        self.word_limit = word_limit
        self.length_limit = length_limit
        self.lang = LangCN()
        self.tokenizer = tokenizer

    def tokenize(self, text_or_words, is_reference=False):
        """
        reference:
        https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L1820
        """
        words = text_or_words
        # tokenization
        if not isinstance(words, list):
            if self.tokenizer:
                words = self.tokenizer.tokenize(text_or_words)
            else:
                words = self.lang.tokenize(text_or_words)

        words = [w.strip().lower() for w in words if w.strip()]

        # limit length
        if self.word_limit > 0:
            words = words[:self.word_limit]
        elif self.length_limit > 0:
            _words = []
            length = 0
            for w in words:
                if length + len(w) < self.length_limit:
                    _words.append(w)
                else:
                    break
            words = _words

        if self.stopwords:
            words = [w for w in words if not self.lang.is_stop_word(w)]

        if self.stemming and is_reference:
            # stemming is only adopted to reference
            # https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L1416

            # min_length ref: https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L2629
            words = [self.lang.stemming(w, min_length=3) for w in words]

        return words

    def parse_to_be(self, text, is_reference=False):
        bes = self.lang.parse_to_be(text)

        def preprocess(be):
            be.head = be.head.lower().strip()
            be.modifier = be.modifier.lower().strip()
            if self.stemming and is_reference:
                be.head = self.lang.stemming(be.head, min_length=3)
                be.modifier = self.lang.stemming(be.modifier, min_length=3)

            return be

        bes = [preprocess(be) for be in bes]
        return bes

    def len_ngram(self, words, n):
        return max(len(words) - n + 1, 0)

    def ngram_iter(self, words, n):
        for i in range(self.len_ngram(words, n)):
            n_gram = words[i:i+n]
            yield tuple(n_gram)

    def count_ngrams(self, words, n):
        c = Counter(self.ngram_iter(words, n))
        return c

    def count_overlap(self, summary_ngrams, reference_ngrams):
        result = 0
        for k, v in summary_ngrams.items():
            result += min(v, reference_ngrams[k])
        return result

    def rouge_1(self, summary, references, alpha=0.5):
        return self.rouge_n(summary, references, 1, alpha)

    def rouge_2(self, summary, references, alpha=0.5):
        return self.rouge_n(summary, references, 2, alpha)

    def rouge_n(self, summary, reference, n, alpha=0.5):
        """
        alpha: alpha -> 0: recall is more important
            alpha -> 1: precision is more important
            F = 1/(alpha * (1/P) + (1 - alpha) * (1/R))
        """
        _summary = self.tokenize(summary)
        summary_ngrams = self.count_ngrams(_summary, n)
        _refs = reference if isinstance(reference, list) else [reference]
        matches = 0
        count_for_recall = 0
        for r in _refs:
            _r = self.tokenize(r, True)
            r_ngrams = self.count_ngrams(_r, n)
            matches += self.count_overlap(summary_ngrams, r_ngrams)
            count_for_recall += self.len_ngram(_r, n)
        count_for_prec = len(_refs) * self.len_ngram(_summary, n)
        f1 = self._calc_f1(matches, count_for_recall, count_for_prec, alpha)
        return f1

    def _calc_f1(self, matches, count_for_recall, count_for_precision, alpha):
        def safe_div(x1, x2):
            return 0 if x2 == 0 else x1 / x2
        recall = safe_div(matches, count_for_recall)
        precision = safe_div(matches, count_for_precision)
        denom = (1.0 - alpha) * precision + alpha * recall
        return safe_div(precision * recall, denom)

    def lcs(self, a, b):
        longer = a
        base = b
        if len(longer) < len(base):
            longer, base = base, longer

        if len(base) == 0:
            return 0

        row = [0] * len(base)
        for c_a in longer:
            left = 0
            upper_left = 0
            for i, c_b in enumerate(base):
                up = row[i]
                if c_a == c_b:
                    value = upper_left + 1
                else:
                    value = max(left, up)
                row[i] = value
                left = value
                upper_left = up

        return left

    def rouge_l(self, summary, reference, alpha=0.5):
        matches = 0
        count_for_recall = 0
        _summary = self.tokenize(summary)
        _refs = reference if isinstance(reference, list) else [reference]
        for r in _refs:
            _r = self.tokenize(r, True)
            matches += self.lcs(_r, _summary)
            count_for_recall += len(_r)
        count_for_prec = len(_refs) * len(_summary)
        f1 = self._calc_f1(matches, count_for_recall, count_for_prec, alpha)
        return f1

    def count_be(self, text, compare_type, is_reference=False):
        bes = self.parse_to_be(text, is_reference)
        be_keys = [be.as_key(compare_type) for be in bes]
        c = Counter(be_keys)
        return c

    def rouge_be(self, summary, references, compare_type="HMR", alpha=0.5):
        matches = 0
        count_for_recall = 0
        s_bes = self.count_be(summary, compare_type)
        _refs = references if isinstance(references, list) else [references]
        for r in _refs:
            r_bes = self.count_be(r, compare_type, True)
            matches += self.count_overlap(s_bes, r_bes)
            count_for_recall += sum(r_bes.values())
        count_for_prec = len(_refs) * sum(s_bes.values())
        f1 = self._calc_f1(matches, count_for_recall, count_for_prec, alpha)
        return f1


class LangCN(BaseLang):

    def __init__(self):
        super(LangCN, self).__init__("cn")

    def tokenize(self, text):
        _txt = self._format_text(text)
        words = _txt.split(" ")
        words = [w.strip() for w in words if w.strip()]
        return words

    def _format_text(self, text):
        _txt = text.strip()
        return _txt

    def parse_to_be(self, text):
        _txt = self._format_text(text)
        return _txt


rouge = RougeCalculator(stopwords=True)


def load_textfiles(reference, hypothesis):
    hypo = {idx: lines.strip() for (idx, lines) in enumerate(hypothesis)}
    # take out newlines before creating dictionary
    refs = {idx: rr.strip() for idx, rr in enumerate(reference)}
    # sanity check that we have the same number of references as hypothesis
    if len(hypo) != len(refs):
        raise ValueError(
            "There is a sentence number mismatch between the inputs")
    return refs, hypo


def calc_score(candidate, refs, segment=True):
    """
    Compute ROUGE-L score given one candidate and references for an image
    :param candidate: str : candidate sentence to be evaluated
    :param refs: list of str : COCO reference sentences for the particular image to be evaluated
    :returns score: int (ROUGE-L score for the candidate evaluated against references)
    """
    # split into tokens
    if not segment:
        token_h = text2charlist(candidate, keep_word="[UNK]")
        summary = " ".join(token_h)
    else:
        summary = candidate

    # split into tokens
    if segment:
        reference = refs
    else:
        token_r = text2charlist(refs, keep_word="[UNK]")
        token_r = filter(lambda x: x != " ", token_r)
        reference = ' '.join(refs)
    # compute the longest common subsequence

    rouge_1 = rouge.rouge_n(
        summary=summary,
        reference=reference,
        n=1
    )

    rouge_2 = rouge.rouge_n(
        summary=summary,
        reference=reference,
        n=2
    )

    rouge_l = rouge.rouge_l(
        summary=summary,
        reference=reference,
    )

    return rouge_1, rouge_2, rouge_l


def score(refs, hypos, segment=True):
    assert(refs.keys() == hypos.keys())
    imgIds = refs.keys()
    rouge_1_l = []
    rouge_2_l = []
    rouge_l_l = []

    for imgid in imgIds:
        hypo = hypos[imgid]
        ref = refs[imgid]

        rouge_1, rouge_2, rouge_l = calc_score(hypo, ref, segment)
        rouge_1_l.append(rouge_1)
        rouge_2_l.append(rouge_2)
        rouge_l_l.append(rouge_l)

    return np.mean(np.array(rouge_1_l)), np.mean(np.array(rouge_2_l)), np.mean(np.array(rouge_l_l))


def calc_rouge_from_file():
    # Feed in the directory where the hypothesis summary and true summary is
    # stored
    segment = True
    reference_file = glob.glob('hypothesis/*')
    hypothesis_file = glob.glob('reference/*')
    assert len(reference_file) == len(hypothesis_file) == 1

    with open(reference_file, 'r', 'utf-8') as rf:
        reference = rf.readlines()

    with open(hypothesis_file, 'r', 'utf-8') as hf:
        hypothesis = hf.readlines()

    ref, hypo = load_textfiles(reference, hypothesis)
    rouge_1, rouge_2, rouge_l = score(ref, hypo, segment)

    print('The average rouge_1 is %s' % rouge_1)
    print('The average rouge_2 is %s' % rouge_2)
    print('The average rouge_l is %s' % rouge_l)


def calc_rouge(ref, hypo, segment=True, _print=False):
    ref, hypo = load_textfiles(ref, hypo)
    rouge_1, rouge_2, rouge_l = score(ref, hypo, segment)

    if _print:
        print('The average rouge_1 is %s' % rouge_1)
        print('The average rouge_2 is %s' % rouge_2)
        print('The average rouge_l is %s' % rouge_l)
    return rouge_1, rouge_2, rouge_l
