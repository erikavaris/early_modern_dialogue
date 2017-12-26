# Early Modern English dialogue generation, by Erika Varis Doggett
# Based on TensorFlow Sequence2Sequence tutorial, for English-French translation.

# Python 3
# ==============================================================================

"""Utilities for parsing Early Modern Dialogue data, tokenizing, vocabularies."""

import os
import re
import tarfile

from six.moves import urllib

from tensorflow.python.platform import gfile
import tensorflow as tf

from nltk.tokenize import word_tokenize
import json
import data_prep
import random

# Special vocabulary symbols - we always put them at the start.
_PAD = b"_PAD"
_GO = b"_GO"
_EOS = b"_EOS"
_UNK = b"_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_DIGIT_RE = re.compile(br"\d")


def basic_tokenizer(sentence):
  """Very basic tokenizer: split the sentence into a list of tokens."""
  words = []
  for space_separated_fragment in sentence.strip().split():
    words.extend(_WORD_SPLIT.split(space_separated_fragment))
  return [w for w in words if w]


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      normalize_digits=True):
  """Create vocabulary file (if it does not exist yet) from data file.

  Data file is assumed to contain one json dict per line, with sentence
  in 'text' key. Each sentence is
  tokenized and digits are normalized (if normalize_digits is set).
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with gfile.GFile(data_path, mode="rb") as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        line = tf.compat.as_str(line)
        line_d = json.loads(line)
        text = line_d['text']
        tokens = word_tokenize(text)
        for w in tokens:
          w = w.encode('utf-8') #I guess I'm using bytes :-P
          word = _DIGIT_RE.sub(b"0", w) if normalize_digits else w
          if word in vocab:
            vocab[word] += 1
          else:
            vocab[word] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with gfile.GFile(vocabulary_path, mode="wb") as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")


def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-json-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary,
                          normalize_digits=True):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "I have a dog" may become tokenized into
  ["I", "have", "a", "dog"] and with vocabulary {"I": 1, "have": 2,
  "a": 4, "dog": 7"} this function will return [1, 2, 4, 7].

  Args:
    sentence: the sentence in bytes format to convert to token-ids.
    vocabulary: a dictionary mapping tokens to integers.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  words = word_tokenize(sentence)
  words = [w.encode('utf-8') for w in words]
  
  if not normalize_digits:
    return [vocabulary.get(w, UNK_ID) for w in words]
  # Normalize digits by 0 before looking words up in the vocabulary.
  return [vocabulary.get(_DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      normalize_digits=True):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
    tokenizer: a function to use to tokenize each sentence;
      if None, basic_tokenizer will be used.
    normalize_digits: Boolean; if true, all digits are replaced by 0s.
  """
  if not gfile.Exists(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with gfile.GFile(data_path, mode="r") as data_file:
      with gfile.GFile(target_path, mode="w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          line = tf.compat.as_str(line)
          sentence = json.loads(line)['text']
          token_ids = sentence_to_token_ids(sentence, vocab,
                                            normalize_digits)
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def prepare_emd_data(data_dir, vocabulary_size=55000):
  """Get Early Modern Dialogue data into data_dir, create vocabularies and tokenize data.

  Args:
    data_dir: directory in which the data sets will be stored.
    en_vocabulary_size: size of the English vocabulary to create and use.
    fr_vocabulary_size: size of the French vocabulary to create and use.
    tokenizer: a function to use to tokenize each data sentence;
      if None, basic_tokenizer will be used.

  Returns:
    A tuple of 6 elements:
      (1) path to the token-ids for input training data-set,
      (2) path to the token-ids for output training data-set,
      (3) path to the token-ids for input development data-set,
      (4) path to the token-ids for output development data-set,
      (5) path to the input vocabulary file,
      (6) path to the output vocabulary file.
  """
  # read and prepare emd data from txt files
  if not gfile.Exists(data_dir+'/input_data.json') and not gfile.Exists(data_dir+'/output_data.json'):
      dialogue_pairs = data_prep.read_ced(data_dir)
      dialogue_pairs.extend(data_prep.read_shakespeare(data_dir))
      data_prep.write_datafiles(dialogue_pairs)
      input_data = [pair[0] for pair in dialogue_pairs]
      output_data = [pair[1] for pair in dialogue_pairs]
  else:
    #read data files
    with open(data_dir+'/input_data.json', 'r') as f:
        input_data = [json.loads(line) for line in f]
    with open(data_dir+'/output_data.json', 'r') as f:
        output_data = [json.loads(line) for line in f]

  indices = range(0, len(input_data))
  dev = random.sample(indices, round(len(input_data)*.1))
  with open(data_dir+'/input_data_training.json', 'w') as from_train:
    with open(data_dir+'/output_data_training.json', 'w') as to_train:
        with open(data_dir+'/input_data_dev.json', 'w') as from_dev:
            with open(data_dir+'/output_data_dev.json', 'w') as to_dev:
                for i in indices:
                    if i not in dev:
                        json.dump(input_data[i], from_train)
                        from_train.write('\n')
                        json.dump(output_data[i], to_train)
                        to_train.write('\n')
                    else:
                        json.dump(input_data[i], from_dev)
                        from_dev.write('\n')
                        json.dump(output_data[i], to_dev)
                        to_dev.write('\n')

  # Get emd data to the specified directory.
  from_train_path = data_dir+'/input_data_training.json'
  to_train_path = data_dir+'/output_data_training.json'
  from_dev_path = data_dir+'/input_data_dev.json'
  to_dev_path = data_dir+'/output_data_dev.json'

  return prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, vocabulary_size=vocabulary_size)


def prepare_data(data_dir, from_train_path, to_train_path, from_dev_path, to_dev_path, vocabulary_size=55000):
  """Preapre all necessary files that are required for the training.

    Args:
      data_dir: directory in which the data sets will be stored.
      from_train_path: path to the file that includes "from" training samples.
      to_train_path: path to the file that includes "to" training samples.
      from_dev_path: path to the file that includes "from" dev samples.
      to_dev_path: path to the file that includes "to" dev samples.
      from_vocabulary_size: size of the "from language" vocabulary to create and use.
      to_vocabulary_size: size of the "to language" vocabulary to create and use.
      tokenizer: a function to use to tokenize each data sentence;
        if None, basic_tokenizer will be used.

    Returns:
      A tuple of 6 elements:
        (1) path to the token-ids for "from language" training data-set,
        (2) path to the token-ids for "to language" training data-set,
        (3) path to the token-ids for "from language" development data-set,
        (4) path to the token-ids for "to language" development data-set,
        (5) path to the "from language" vocabulary file,
        (6) path to the "to language" vocabulary file.
    """
  # Create vocabularies of the appropriate sizes.
  # for same language dialogue system, use the same vocab size
  to_vocabulary_size = vocabulary_size
  from_vocabulary_size = vocabulary_size

  to_vocab_path = os.path.join(data_dir, "vocab%d.to" % to_vocabulary_size)
  from_vocab_path = os.path.join(data_dir, "vocab%d.from" % from_vocabulary_size)
  create_vocabulary(to_vocab_path, to_train_path , to_vocabulary_size)
  create_vocabulary(from_vocab_path, from_train_path , from_vocabulary_size)

  # Create token ids for the training data.
  to_train_ids_path = to_train_path + (".ids%d" % to_vocabulary_size)
  from_train_ids_path = from_train_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_train_path, to_train_ids_path, to_vocab_path)
  data_to_token_ids(from_train_path, from_train_ids_path, from_vocab_path)

  # Create token ids for the development data.
  to_dev_ids_path = to_dev_path + (".ids%d" % to_vocabulary_size)
  from_dev_ids_path = from_dev_path + (".ids%d" % from_vocabulary_size)
  data_to_token_ids(to_dev_path, to_dev_ids_path, to_vocab_path)
  data_to_token_ids(from_dev_path, from_dev_ids_path, from_vocab_path)

  return (from_train_ids_path, to_train_ids_path,
          from_dev_ids_path, to_dev_ids_path,
          from_vocab_path, to_vocab_path)
