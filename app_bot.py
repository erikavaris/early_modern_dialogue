import tensorflow as tf
from app_config import Configuration
import os
import logging
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
import numpy as np

import data_utils
import seq2seq_model

_buckets = [(7,8), (16,16), (25,24), (46,50)] # buckets manually identified after examining data

class ShakespeareBot(object):

    def __init__(self):

        # attempt at pre-loading the model
        logging.info("Loading the dialogue bot model now...")
        graph = tf.Graph()
        with graph.as_default():
            self.sess = tf.Session() 
            # Create model and load parameters.
            self.model = self.create_model(self.sess, True)
            self.model.batch_size = 1  # We decode one sentence at a time.

            # Load vocabularies.
            from_vocab_path = os.path.join(Configuration.DATA_DIR,
                                         "vocab%d.from" % Configuration.VOCAB_SIZE)
            to_vocab_path = os.path.join(Configuration.DATA_DIR,
                                         "vocab%d.to" % Configuration.VOCAB_SIZE)
            self.from_vocab, _ = data_utils.initialize_vocabulary(from_vocab_path)
            _, self.rev_to_vocab = data_utils.initialize_vocabulary(to_vocab_path)

    def create_model(self, session, forward_only):
        """Create dialogue model and initialize or load parameters in session."""
        dtype = tf.float32
        model = seq2seq_model.Seq2SeqModel(
          Configuration.VOCAB_SIZE,
          _buckets,
          Configuration.SIZE,
          Configuration.NUM_LAYERS,
          Configuration.MAX_GRADIENT_NORM,
          Configuration.BATCH_SIZE,
          Configuration.LEARNING_RATE,
          Configuration.LEARNING_RATE_DECAY_FACTOR,
          forward_only=forward_only,
          dtype=dtype)
        # use existing model & checkpoint
        ckpt = tf.train.get_checkpoint_state(Configuration.TRAIN_DIR)
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
        return model


    def respond(self, sentence):
        logging.info("Analyzing input sentence for response...")  
        # Get token-ids for the input sentence.
        token_ids = data_utils.sentence_to_token_ids(tf.compat.as_str(sentence), self.from_vocab)
        # Which bucket does it belong to?
        bucket_id = len(_buckets) - 1
        for i, bucket in enumerate(_buckets):
            if bucket[0] >= len(token_ids):
                bucket_id = i
                break
            else:
                logging.warning("Sentence truncated: %s", sentence)

        # Get a 1-element batch to feed the sentence to the model.
        encoder_inputs, decoder_inputs, target_weights = self.model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
        # Get output logits for the sentence.
        _, _, output_logits = self.model.step(self.sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
        # This is a greedy decoder - outputs are just argmaxes of output_logits.
        outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
        # If there is an EOS symbol in outputs, cut them at that point.
        if data_utils.EOS_ID in outputs:
            outputs = outputs[:outputs.index(data_utils.EOS_ID)]
        # Return model-generated sentence corresponding to outputs.
        return " ".join([tf.compat.as_str(self.rev_to_vocab[output]) for output in outputs])
