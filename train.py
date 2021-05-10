#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import pickle
import time
from jieba import cut
import datetime
import data_helpers
from cnn_model import TextCNN
#from model_v2 import TextCNN
from tensorflow.contrib import learn
import sklearn
from sklearn.cluster import KMeans
import random
#from gensim.models.keyedvectors import KeyedVectors
#word2vec_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

# Parameters
# ==================================================

# Data loading params
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("dataset","MR","directory of dataset: SST-5/MR/ARSC/yelp/twitter")
tf.flags.DEFINE_string("positive_data_file", "../data/MR_polarity_5k/rt-polarity.pos", "Data source for the MR positive data.")
tf.flags.DEFINE_string("negative_data_file", "../data/MR_polarity_5k/rt-polarity.neg", "Data source for the MR negative data.")
tf.flags.DEFINE_float("initialize_range", 0.2, "initialize range of word embedding")
tf.flags.DEFINE_integer("max_sentence_length", 100, "Max sentence length in train/test data (Default: 100)")
tf.flags.DEFINE_string("word2vec", '../CNN-text-classification/GoogleNews-vectors-negative300.bin', "Word2vec file with pre-trained embeddings is GoogleNews-vectors-negative300.bin. (Default: None)")
tf.flags.DEFINE_string("positive_lexicon_file", "./new_positive_lexicon.txt", "positive lexicon file")
tf.flags.DEFINE_string("negative_lexicon_file", "./new_negative_lexicon.txt", "negative lexicon file")
tf.flags.DEFINE_string("negation_lexicon_file", "./negation_lexicon.txt", "negation lexicon file")
tf.flags.DEFINE_string("intensifier_lexicon_file", "./intensifier_lexicon.txt", "intensifier lexicon file")
tf.flags.DEFINE_integer("cut_train_data", None, "reduce train data to the size (Default: None)")

# Model Hyperparameters
tf.flags.DEFINE_string("channel_setting", "both", "both/knowledge/general (default: 'both')")
tf.flags.DEFINE_integer("num_clusters", 100, "num_clusters (default: 100)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (default: 300)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 30, "Number of filters per filter size (default: 100)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3, "L2 regularization lambda (default: 0.0)")
#tf.flags.DEFINE_string("model_type","rand","'rand' for CNN-rand; 'static' for CNN-static (default: rand)")
# Training parameters
tf.flags.DEFINE_integer("batch_size", 50, "Batch Size (default: 50)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 6, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# print("\nParameters:")
# for attr, value in sorted(FLAGS.__flags.items()):
#     print("{}={}".format(attr.upper(), value))
# print("")
def chinese_tokenizer(docs):
    for doc in docs:
        yield list(cut(doc))

def preprocess():
    print("Loading lexicons...")
    pos_list, neg_list, negation_list, intensifier_list = data_helpers.load_lexicon(FLAGS.positive_lexicon_file, FLAGS.negative_lexicon_file, FLAGS.negation_lexicon_file, FLAGS.intensifier_lexicon_file)
    lexicon_list_raw = pos_list + neg_list + negation_list + intensifier_list
    
    print("Building Vocabulary Processor...")
    if FLAGS.dataset == 'MR':
        x_text, y = data_helpers.load_MR(FLAGS.positive_data_file, FLAGS.negative_data_file)
        
    if FLAGS.dataset == 'SST-5':
        train_text, train_label, dev_text, dev_label, test_text, test_label = data_helpers.load_SST5('../data/SST-5')
        x_text = train_text + dev_text + test_text
        y = train_label + dev_label + test_label
        

    if FLAGS.dataset == 'ARSC':
        x_text, y = data_helpers.load_ARSC('../data/ARSC')
        
    if FLAGS.dataset == 'yelp':
        x_text, y = data_helpers.load_yelp('../data/yelp')
        
    if FLAGS.dataset == 'twitter':
        x_text, y = data_helpers.load_twitter('../data/twitter')
        
    vocab_processor = learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)
    vocab_processor.fit_transform(x_text + lexicon_list_raw)
    x = np.array(list(vocab_processor.transform(x_text)))
    y = np.array(y)
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))


    # initial matrix with random uniform
    initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
    # load any vectors from the word2vec
    
    print("Load word2vec file {0}".format(FLAGS.word2vec))
    with open(FLAGS.word2vec, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in range(vocab_size):
            word = []
            while True:
                ch = f.read(1).decode('latin-1')
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab_processor.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)
    print("Success to load pre-trained word2vec model!\n")
    

    print("Processing filters...")
    if FLAGS.channel_setting == "general":
        pos_list=None
        neg_list=None
        negation_list=None
        intensifier_list=None
    else:
        def process_list(input_list):
            input_list = list(vocab_processor.transform(input_list))
            filter1 = []
            filter2 = []
            filter3 = []
            for item in input_list:
                item = list(item)
                if len(item) - item.count(0) == 1:
                    filter1.append(item)
                elif len(item) - item.count(0) == 2:
                    filter2.append(item)
                elif len(item) - item.count(0) == 3:
                    filter3.append(item)
            if len(filter1) != 0:
                filter1 = np.squeeze(np.array(filter1)[:, 0:1])
            if len(filter2) != 0:
                filter2 = np.array(filter2)[:, 0:2]
            if len(filter3) != 0:
                filter3 = np.array(filter3)[:, 0:3]
            return [filter1, filter2, filter3]

        pos_list = process_list(pos_list)
        neg_list = process_list(neg_list)
        negation_list = process_list(negation_list)
        intensifier_list = process_list(intensifier_list)

        #clustering
        
        def filter_clustering(input_filter_list, num_clusters):
            filter_embedding = np.array(initW[input_filter_list])
            if len(filter_embedding.shape)==3:
                filter_embedding = np.squeeze(np.concatenate([filter_embedding[:, 0:1, :], filter_embedding[:, 1:2, :]], axis=-1), axis=1)
            print("\nfilter_clustering / filter_embedding shape: ", filter_embedding.shape)
            kmeans = KMeans(n_clusters=num_clusters, max_iter=300, n_init=40, init='k-means++',n_jobs=-1).fit(filter_embedding)
            cluster_labels = list(zip(input_filter_list, list(kmeans.labels_)))
            cluster_labels = sorted(cluster_labels, key=lambda x:x[1])
            #print("cluster_labels\n: ", cluster_labels)
            new_filter_list = [t for (t, l) in cluster_labels]
            filter_label_list = [l for (t, l) in cluster_labels]
            filter_count = [filter_label_list.count(idx) for idx in range(0, num_clusters)]
            #print("filter_clustering / length of new_filter_list: ", len(new_filter_list))
            print("filter_clustering / filter_count: ", filter_count)
            return [np.array(new_filter_list), filter_count]

    
        print("K-means clustering...")
        pos_list = [filter_clustering(pos_list[0], FLAGS.num_clusters), pos_list[1], pos_list[2]]
        neg_list = [filter_clustering(neg_list[0], FLAGS.num_clusters), neg_list[1], neg_list[2]]
        #negation_list = filters[2]
        #intensifier_list = filters[3]
        print("after clustering:\n")
        print("pos_list\n", pos_list)
        print("neg_list\n", neg_list)
        print("negation_list\n", negation_list)
        print("intensifier_list\n", intensifier_list)

    
    filters = [pos_list, neg_list, negation_list, intensifier_list]
    print("len(x), len(y): ", len(x), len(y))
    return vocab_processor, initW, x, y, filters

def dataprocess(x, y):
    
    # Data Preparation
    # ==================================================
    print("Shuffling data...")
    
    # Randomly shuffle data
    np.random.seed()
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_test = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_test = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y_train)))
    x_train, x_dev = x_train[:dev_sample_index], x_train[dev_sample_index:]
    y_train, y_dev = y_train[:dev_sample_index], y_train[dev_sample_index:]

    #cut train data
    if FLAGS.cut_train_data != None:
        x_train = x_train[:FLAGS.cut_train_data]
        y_train = y_train[:FLAGS.cut_train_data]

        
    del x, y, x_shuffled, y_shuffled
    print("top 3 of training samples: \n", x_train[0:3])
    print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), len(y_dev), len(y_test)))

    return x_train, y_train, x_dev, y_dev, x_test, y_test

def train(timestamp, round_num, x_train, y_train, vocab_processor, x_dev, y_dev, filters, initW):
    # Training
    print("Training round {:d} \n".format(round_num))
    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                channel_setting=FLAGS.channel_setting,
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                pos_list=filters[0],
                neg_list=filters[1],
                negation_list=filters[2], 
                intensifier_list=filters[3],
                initW=initW)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            #timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp, "round_"+str(round_num)))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))
            

            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            tvars = tf.trainable_variables()
            print("trainable variables:\n")
            for item in tvars:
                print(item)

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                
                time_str = datetime.datetime.now().isoformat()
                #print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy= sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy
            
            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            max_dev_acc=-1
            saved_steps = []
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_acc=dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    if dev_acc > max_dev_acc:
                        max_dev_acc = dev_acc
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        saved_steps.append(current_step)
                        print("Saved model checkpoint to {}\n".format(path))
            path = saver.save(sess, checkpoint_prefix, global_step=current_step)
            saved_steps.append(current_step)
            print("Last saved model checkpoint to {}\n".format(path))
            return checkpoint_dir, saved_steps[-FLAGS.num_checkpoints:]

def eval(round_num, checkpoint_dir, saved_steps, x_test, y_test, vocab_processor):
    y_test = np.argmax(y_test, axis=1).astype(int)
    print("Evaluating round {:d}...".format(round_num))
    print("Total number of test examples: {}".format(len(y_test)))
    

    eval_acc = -1
    eval_step = -1
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            for step in saved_steps:
                checkpoint_file = checkpoint_dir+"/model-"+str(step)
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
            
                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                # input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                # Generate batches for one epoch
                batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)
                
                # Collect the predictions here
                all_predictions = []

                for x_test_batch in batches:
                    batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_predictions = np.concatenate([all_predictions, batch_predictions])
                # Print accuracy if y_test is defined
                all_predictions = np.array(all_predictions).astype(int)
                correct_predictions = all_predictions == y_test
                correct_predictions = float(sum(correct_predictions))
                acc = correct_predictions/float(len(y_test))
                print("model-{:d} / Accuracy: {:g}".format(step, acc))
                
                if acc > eval_acc:
                    eval_acc = acc
                    eval_step = step
                    
    return eval_acc, eval_step


def main(argv=None):
    assert FLAGS.word2vec != None
    if FLAGS.dataset=='MR':
        assert FLAGS.positive_data_file != None
        assert FLAGS.negative_data_file != None
    
    cut_train_datas = [500, None]
    #cluster_nums = [50, 100, 150]
    #filter_nums = [30, 100]
    datasets = ['MR', 'SST-5']


    answers = []
    for dt in datasets:
        FLAGS.dataset = dt
        for ct in cut_train_datas:
            FLAGS.cut_train_data = ct
            if dt=='MR' and ct==None:
                continue
            print("Start Tuning: dataset={}, cut_train_data={}".format(FLAGS.dataset, FLAGS.cut_train_data))
            vocab_processor, initW, x, y, filters = preprocess()
            #results = []
            eval_acc_res = []
            timestamp = str(int(time.time()))
            total_time = 0
            for round_num in range(0,3):
                x_train, y_train, x_dev, y_dev, x_test, y_test = dataprocess(x, y)
                start_time = int(time.time())
                checkpoint_dir, saved_steps = train(timestamp, round_num, x_train, y_train, vocab_processor, x_dev, y_dev, filters, initW)
                end_time = int(time.time())
                total_time += end_time - start_time
                eval_acc, eval_step = eval(round_num, checkpoint_dir, saved_steps, x_test, y_test, vocab_processor)
                #results.append((round_num, eval_step, eval_acc))
                eval_acc_res.append(eval_acc)

            #print("final results: \n", results)
            ave = sum(eval_acc_res) / len(eval_acc_res)
            std = np.std(eval_acc_res)
            print("dataset: {}, cut_train_data: {}, accuracy: {:g}, std: {:g}, time: {:g}".format(FLAGS.dataset, FLAGS.cut_train_data, ave, std, total_time/3))
            answers.append((FLAGS.dataset, FLAGS.cut_train_data, ave, std, total_time/3))

    for (dt, ct, ave, std, t) in answers:
        print("dataset {}, cut_train_data: {}, accuracy: {:g}, std: {:g}, time: {:g} \n".format(dt, ct, ave, std, t))

    res_path = os.path.join(checkpoint_dir, "..", "..", "all_rounds_results.txt")
    with open(res_path, 'w') as w:
        #w.write("Writing to " + timestamp + "\n")
        for (dt, ct, ave, std, t) in answers:
            w.write("dataset {}, cut_train_data: {}, accuracy: {:g}, std: {:g}, time: {:g} \n".format(dt, ct, ave, std, t))

if __name__ == '__main__':
    tf.app.run()