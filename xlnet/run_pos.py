from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
import os
import numpy as np
import random
import tensorflow as tf
import sentencepiece as spm

from data_utils import SEP_ID, VOCAB_SIZE, CLS_ID
import model_utils
from classifier_utils import PaddingInputExample
from classifier_utils import convert_single_example
from prepro_utils import preprocess_text, encode_ids

import xlnet

# Model
flags.DEFINE_string("model_config_path", default=None,
                    help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,
                   help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,
                   help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,
                     help="Clamp length")
flags.DEFINE_string("summary_type", default="last",
                    help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True,
                  help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", default=False,
                  help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",
                  enum_values=["normal", "uniform"],
                  help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,
                   help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,
                   help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False,
                  help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default=None,
                    help="checkpoint path for initializing the model. "
                         "Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("output_dir", default="",
                    help="Output dir for TF records.")
flags.DEFINE_string("spiece_model_file", default="",
                    help="Sentence Piece model path.")
flags.DEFINE_string("model_dir", default="",
                    help="Directory for saving the finetuned model.")
flags.DEFINE_string("data_dir", default="",
                    help="Directory of data.")
flags.DEFINE_string("train_file", default="",
                    help="Training file.")
flags.DEFINE_string("test_file", default="",
                    help="Testing file.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,
                     help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context "
                          "of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=1000,
                     help="number of iterations per TPU training loop.")

# Training
flags.DEFINE_bool("do_train", default=False, help="whether to do training")
flags.DEFINE_integer("train_steps", default=12000,
                     help="Number of training steps")
flags.DEFINE_integer("num_train_epochs", default=3,
                     help='Number of training epochs')
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=2e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,
                   "Top layer: lr[L] = FLAGS.learning_rate."
                   "Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,
                   help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=3,
                     help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("save_steps", default=None,
                     help="Save the model for every save_steps. "
                          "If None, not to save any model.")
flags.DEFINE_integer("train_batch_size", default=8,
                     help="Batch size for training. Note that batch size 1 corresponds to "
                          "4 sequences: one paragraph + one quesetion + 4 candidate answers.")
flags.DEFINE_float("weight_decay", default=0.00, help="weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-6, help="adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")

# Evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_string("eval_split", default="dev",
                    help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=32,
                     help="Batch size for evaluation.")

# Data config
flags.DEFINE_integer("max_seq_length", default=512,
                     help="Max length for the paragraph.")
flags.DEFINE_integer("shuffle_buffer", default=2048,
                     help="Buffer size used for shuffle.")
flags.DEFINE_bool("uncased", default=False,
                  help="Use uncased.")

FLAGS = flags.FLAGS

SEG_ID_A = 0
SEG_ID_B = 1
SEG_ID_CLS = 2
SEG_ID_SEP = 3
SEG_ID_PAD = 4


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A signle set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_list,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_list = label_list
        self.is_real_example = is_real_example


class InputExample(object):
    def __init__(self, id, sent, label_list):
        """
        :param id: Unique id for the example
        :param sent: string. The untokenized text of the sentence.
        :param label_list: list. Labels for all words.
        """

        self.id = id
        self.sent = sent
        self.label_list = label_list


def convert_single_example(example, tokenize_fn, all_labels):
    """Converts a single `InputExample` into a signle `InputFeatures`.'"""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * FLAGS.max_seq_length,
            input_mask=[1] * FLAGS.max_seq_length,
            segment_ids=[0] * FLAGS.max_seq_length,
            label_list=[0] * FLAGS.max_seq_length,
            is_real_example=False
        )

    tokens = []
    segment_ids = []
    label_list = []
    input_mask = []

    for i, word in enumerate(example.sent):
        tokens_word = tokenize_fn(word)
        if len(tokens_word) == 1:
            tokens.append(tokens_word[0])
            segment_ids.append(SEG_ID_A)
            label_list.append(all_labels.index(example.label_list[i]))
            input_mask.append(0)
        else:
            j = 0
            for token in tokens_word:
                tokens.append(token)
                segment_ids.append(SEG_ID_A)
                if j == 0:
                    label_list.append(all_labels.index(example.label_list[i]))
                    input_mask.append(0)
                    j = 1
                else:
                    label_list.append(all_labels.index('##'))
                    input_mask.append(1)

    if len(tokens) >= FLAGS.max_seq_length:
        tokens = tokens[ : FLAGS.max_seq_length - 2]
        input_mask = input_mask[: FLAGS.max_seq_length - 2]
        segment_ids = segment_ids[: FLAGS.max_seq_length - 2]
        label_list = label_list[: FLAGS.max_seq_length - 2]

    tokens.append(SEP_ID)
    segment_ids.append(SEG_ID_A)
    input_mask.append(1)
    label_list.append('PAD')

    tokens.append(CLS_ID)
    segment_ids.append(SEG_ID_CLS)
    input_mask.append(1)
    label_list.append('PAD')

    input_ids = tokens
    if len(input_ids) < FLAGS.max_seq_length:
        delta_len = FLAGS.max_seq_length - len(input_ids)
        input_ids = [0] * delta_len + input_ids
        input_mask = [1] * delta_len + input_mask
        segment_ids = [SEG_ID_PAD] * delta_len + segment_ids
        label_list = [all_labels.index('PAD')] * delta_len + label_list

    assert len(input_ids) == FLAGS.max_seq_length
    assert len(input_mask) == FLAGS.max_seq_length
    assert len(segment_ids) == FLAGS.max_seq_length
    assert len(label_list) == FLAGS.max_seq_length

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_list=label_list,
        is_real_example=True
    )

    return feature

def get_labels(data_dir, mode):
    data_dir = data_dir + '/treebank.' + mode
    labels = set()
    with open(data_dir) as file:
        for line in file.readlines():
            words = line.split(" ")[:-1]
            words = [word.split("/") for word in words]
            label = [x[-1] for x in words]
            for x in label:
                labels.add(x)
    return list(labels)


def conver_examples_to_features(examples, all_labels, tokenize_fn):
    features = []
    for example in examples:
        feature = convert_single_example(example, tokenize_fn, all_labels)
        features.append(feature)
    return features

def create_examples(data_dir, mode):
    max_seq_length = FLAGS.max_seq_length
    data_dir = data_dir + '/treebank.' + mode

    def _read_pos_examples(filename):
        examples = []
        with open(filename) as file:
            i = 0
            for line in file.readlines():
                words = line.split(" ")[:-1]
                words = [word.split("/") for word in words]
                sent = ["".join(x[:-1]) for x in words]
                label = [x[-1] for x in words]
                example = InputExample(id=i, sent=sent, label_list=label)
                i += 1
                examples.append(example)
        return examples

    return _read_pos_examples(data_dir)


def input_fn_builder(features, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to Estimator."""
    seq_length = FLAGS.max_seq_length

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_list = []
    all_is_real_example = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_list.append(feature.label_list)
        all_is_real_example.append(feature.is_real_example)

    def input_fn(params):
        """The actual input function"""
        batch_size = params["batch_size"]

        num_examples = len(features)

        d = tf.data.Dataset.from_tensor_slices({
            "input_ids":
                tf.constant(
                    all_input_ids, shape=[num_examples, seq_length]
                ),
            "input_mask":
                tf.constant(all_input_mask, shape=[num_examples, seq_length]),
            "segment_ids":
                tf.constant(all_segment_ids, shape=[num_examples, seq_length]),
            "label_list":
                tf.constant(all_label_list, shape=[num_examples, seq_length], dtype=tf.int32),
            "is_real_example":
                tf.constant(all_is_real_example, shape=[num_examples], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100, seed=FLAGS.seed)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d
    return input_fn


def create_model(FLAGS, features, is_training, num_labels):
    xlnet_config = xlnet.XLNetConfig(json_path=FLAGS.model_config_path)
    run_config = xlnet.create_run_config(is_training, True, FLAGS)

    inp = features["input_ids"]
    seg_id = features["segment_ids"]
    inp_mask = features["input_mask"]
    label_list = features["label_list"]

    xlnet_model = xlnet.XLNetModel(
        xlnet_config=xlnet_config,
        run_config=run_config,
        input_ids=inp,
        seg_ids=seg_id,
        input_mask=inp_mask)

    output = xlnet_model.get_sequence_output()

    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output, keep_prob=0.9)
        logits = tf.layers.dense(output, num_labels, activation=None)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, num_labels])

        input_mask = tf.cast(inp_mask, dtype=tf.float32)

        log_prob = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(label_list, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_prob, axis=-1)
        input_mask *= -1
        input_mask += 1
        per_example_loss *= input_mask

        loss = tf.reduce_mean(per_example_loss)

    return loss, per_example_loss, logits


def get_model_fn(num_labels):
    def model_fn(features, labels, mode, params):
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        total_loss, per_example_loss, logits = create_model(FLAGS, features, is_training, num_labels)

        num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
        tf.logging.info('#params: {}'.format(num_params))

        scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

        #### Evaluation mode
        if mode == tf.estimator.ModeKeys.EVAL:
            assert FLAGS.num_hosts == 1

            def metric_fn(per_example_loss, label_list, logits, input_mask):
                input_mask *= -1
                input_mask += 1

                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                eval_input_dict = {
                    'labels': label_list,
                    'predictions': predictions,
                    'weights': input_mask
                }
                accuracy = tf.metrics.accuracy(**eval_input_dict)

                loss = tf.metrics.mean(values=per_example_loss, weights=input_mask)
                return {
                    'eval_accuracy': accuracy,
                    'eval_loss': loss}

            input_mask = tf.cast(features['input_mask'], dtype=tf.float32)

            label_list = features['label_list']
            metric_args = [per_example_loss, label_list, logits, input_mask]

            if FLAGS.use_tpu:
                eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=(metric_fn, metric_args),
                    scaffold_fn=scaffold_fn)
            else:
                eval_spec = tf.estimator.EstimatorSpec(
                        mode=mode,
                        loss=total_loss,
                        eval_metric_ops=metric_fn(*metric_args))
            return eval_spec

        #### Configuring the optimizer
        train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

        monitor_dict = {}
        monitor_dict["lr"] = learning_rate

        #### Constucting training TPUEstimatorSpec with new cache.
        if FLAGS.use_tpu:
            #### Creating host calls
            host_call = None

            train_spec = tf.contrib.tpu.TPUEstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
                scaffold_fn=scaffold_fn)
        else:
            train_spec = tf.estimator.EstimatorSpec(
                mode=mode, loss=total_loss, train_op=train_op)

        return train_spec

    return model_fn

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    all_labels = get_labels(FLAGS.data_dir, FLAGS.train_file)
    all_labels.append('##')
    all_labels.append('PAD')

    #### Validate flags
    if FLAGS.save_steps is not None:
        FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError(
            "At least one of `do_train` or `do_eval` must be True.")

    if not tf.gfile.Exists(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.spiece_model_file)

    def tokenize_fn(text):
        text = preprocess_text(text, lower=FLAGS.uncased)
        return encode_ids(sp, text)

        # TPU Configuration

    run_config = model_utils.configure_tpu(FLAGS)

    model_fn = get_model_fn(len(all_labels))

    spm_basename = os.path.basename(FLAGS.spiece_model_file)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    if FLAGS.use_tpu:
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size)
    else:
        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config)



    if FLAGS.do_train:

        train_examples = create_examples(FLAGS.data_dir, FLAGS.train_file)
        random.shuffle(train_examples)

        train_features = conver_examples_to_features(train_examples, all_labels, tokenize_fn)

        num_train_steps = int(
            len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        tf.logging.info('Train steps: '+ str(num_train_steps) + '.')

        train_input_fn = input_fn_builder(
            features=train_features,
            drop_remainder=True,
            is_training=True
        )
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = create_examples(FLAGS.data_dir, FLAGS.test_file)
        while len(eval_examples) % FLAGS.eval_batch_size != 0:
            eval_examples.append(PaddingInputExample())
        assert len(eval_examples) % FLAGS.eval_batch_size == 0
        eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)
        eval_features = conver_examples_to_features(eval_examples, all_labels, tokenize_fn)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = input_fn_builder(
                features=eval_features,
                drop_remainder=eval_drop_remainder,
                is_training=False
            )

        ret = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        tf.logging.info("=" * 80)
        log_str = "Eval | "
        for key, val in ret.items():
            log_str += "{} {} | ".format(key, val)
        tf.logging.info(log_str)
        tf.logging.info("=" * 80)

if __name__ == '__main__':
    tf.app.run()



