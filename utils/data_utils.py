import os, re


class Vocab(object):
  def __init__(self, filename, init=False):
    self.filename = filename
    self.vocab_list = []
    self.vocab_dict = {}
    if os.path.exists(filename) and not init:
      with open(filename, 'r') as f:
        for line in f:
          [key, count] = line.strip("\n").split("\t")
          self.vocab_list.append(key)
          self.vocab_dict[key] = [len(self.vocab_dict), int(count)]
    self.changed = False

  def idx2key(self, idx):
    """given index return key"""
    if idx >= len(self.vocab_list):
      return None
    else:
      return self.vocab_list[idx]

  def key2idx(self, key):
    """given key return index"""
    value = self.vocab_dict.get(key)
    if value:
      return value[0]
    else:
      return None

  def size(self):
    """return size of the vocab"""
    return len(self.vocab_list)

  def dump(self):
    """dump the vocab to the file"""
    if self.changed:
      with open(self.filename, 'w') as f:
        for key in self.vocab_list:
          f.write(key+'\t'+str(self.vocab_dict[key][1])+'\n')

  def update(self, patch):
    """update the vocab"""
    self.changed = True
    for key in patch:
      if self.vocab_dict.has_key(key):
        self.vocab_dict[key][1] += patch[key]
      else:
        self.vocab_dict[key] = [len(self.vocab_dict), patch[key]]
    self.vocab_list = sorted(self.vocab_dict, key=lambda i: self.vocab_dict.get(i)[1], reverse=True)
    for idx in xrange(len(self.vocab_list)):
      self.vocab_dict[self.vocab_list[idx]][0] = idx




def sentence_to_token_ids(text, vocab):
  """encode a sentence in plain text into a sequence of token ids
     token_ids are one-based
  """
  text = text.strip()
  seq = [vocab.key2idx(key.encode('utf8')) for key in list(text.decode('utf8'))]
  seq = [idx+1 if idx else vocab.key2idx("_UNK")+1 for idx in seq]
  return seq

def token_ids_to_sentence(token_ids, vocab):
  """decode a sequence of token ids to a sentence
     token_ids must be one-based
  """
  token_ids = filter(lambda i: i > 0, token_ids)
  token_ids = map(lambda i: i-1 if vocab.idx2key(i-1) else vocab.key2idx("_UNK"), token_ids)
  text = "".join([vocab.idx2key(i) for i in token_ids])
  return text
