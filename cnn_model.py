import tensorflow as tf
import numpy as np

#from tensorflow.python.platform import flags
#FLAGS = flags.FLAGS

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, channel_setting, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda, pos_list, neg_list, negation_list, intensifier_list, initW):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            init = tf.constant_initializer(initW)
            self.W_non_static = tf.get_variable(
                shape=[vocab_size, embedding_size],
                initializer=init,
                name="W_non_static",
                trainable=False)
            self.W_static = tf.get_variable(
                shape=[vocab_size, embedding_size],
                initializer=init,
                name="W_static",
                trainable=False)
            self.word_embedding_static = tf.nn.embedding_lookup(self.W_static, self.input_x)
            self.word_embedding_non_static = tf.nn.embedding_lookup(self.W_non_static, self.input_x)
            
            self.word_embedding_static_expanded = tf.expand_dims(self.word_embedding_static, -1)
            self.word_embedding_non_static_expanded = tf.expand_dims(self.word_embedding_non_static, -1)
            #self.embedded_chars_expanded = tf.concat([self.word_embedding, self.embedded_chars_expanded2], axis=3)

        print("shape of word_embedding_static_expanded: ", self.word_embedding_static_expanded.shape)

        
        # knowledge_channel
        if channel_setting != 'general':
            with tf.name_scope("knowledge_channel"):
                self.word_magnitude_square = word_magnitude_square = tf.reduce_sum(tf.multiply(self.word_embedding_static, self.word_embedding_static), -1, name="word_magnitude_square") # batch_size, seq_len
                print("word_magnitude_square.shape: ", word_magnitude_square.shape)
                
                def filter_conv(input_list, cluster_list, filter_length, v_name):
                    num_filter = len(input_list)
                    print("filter_conv / num_filter: ", num_filter)
                    filter_embedding = tf.nn.embedding_lookup(self.W_static, input_list, name=v_name + "_filter_embedding")
                    print("filter_conv / filter_embedding.shape: ", filter_embedding.shape)
                    if filter_length == 1:
                        full_filters = tf.expand_dims(tf.expand_dims(tf.transpose(filter_embedding),0), 2)
                    else:
                        full_filters = tf.expand_dims(tf.transpose(filter_embedding, [1, 2, 0]), 2)
                    print("filter_conv / full_filters.shape: ", full_filters.shape)
                    
                    conv = tf.nn.conv2d(
                        self.word_embedding_static_expanded,
                        full_filters,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name=v_name + "_conv") #batch_size, seq_len - filter_len + 1, 1, num_filter

                    conv_reduce = tf.squeeze(conv, squeeze_dims=2) #batch_size, seq_len - filter_len + 1, num_filter
                    print("filter_conv / conv_reduce.shape: ", conv_reduce.shape)

                    filter_magnitude_square = tf.reduce_sum(tf.multiply(filter_embedding, filter_embedding), -1, name=v_name+"_filter_magnitude_square") # num_filter or [num_filter,2] 
                    print("filter_conv / filter_magnitude_square.shape: ", filter_magnitude_square.shape)
                    if filter_length == 1:
                        word_magnitude_dividor = tf.tile(tf.expand_dims(tf.sqrt(word_magnitude_square), -1), [1, 1, num_filter], name=v_name+"_word_magnitude_dividor") # batch_size, seq_len, num_filter
                        filter_magnitude_dividor = tf.tile(tf.expand_dims(tf.sqrt(filter_magnitude_square), 0), [sequence_length - filter_length + 1, 1], name=v_name+"_filter_magnitude_dividor") # seq_len, num_filter
                    elif filter_length == 2:
                        word_magnitude_square2 = tf.slice(word_magnitude_square, [0,0], [-1, sequence_length-1]) + tf.slice(word_magnitude_square, [0,1], [-1, sequence_length-1])
                        word_magnitude_dividor = tf.tile(tf.expand_dims(tf.sqrt(word_magnitude_square2), -1), [1, 1, num_filter], name=v_name+"_word_magnitude_dividor")
                        filter_magnitude_square2 = tf.squeeze(tf.slice(filter_magnitude_square, [0,0], [-1, 1]) + tf.slice(filter_magnitude_square, [0,1], [-1, 1]), squeeze_dims=-1) # num_filter
                        filter_magnitude_dividor = tf.tile(tf.expand_dims(tf.sqrt(filter_magnitude_square2), 0), [sequence_length - filter_length + 1, 1], name=v_name+"_filter_magnitude_dividor")
                    elif filter_length == 3:
                        word_magnitude_square3 = tf.slice(word_magnitude_square, [0,0], [-1, sequence_length-2]) + tf.slice(word_magnitude_square, [0,1], [-1, sequence_length-2]) + tf.slice(word_magnitude_square, [0,2], [-1, sequence_length-2])
                        word_magnitude_dividor = tf.tile(tf.expand_dims(tf.sqrt(word_magnitude_square3), -1), [1, 1, num_filter], name=v_name+"_word_magnitude_dividor")
                        filter_magnitude_square3 = tf.squeeze(tf.slice(filter_magnitude_square, [0,0], [-1, 1]) + tf.slice(filter_magnitude_square, [0,1], [-1, 1]) + tf.slice(filter_magnitude_square, [0,2], [-1, 1]), squeeze_dims=-1) # num_filter
                        filter_magnitude_dividor = tf.tile(tf.expand_dims(tf.sqrt(filter_magnitude_square3), 0), [sequence_length - filter_length + 1, 1], name=v_name+"_filter_magnitude_dividor")
                    
                    print("filter_conv / word_magnitude_dividor.shape: ", word_magnitude_dividor.shape)
                    print("filter_conv / filter_magnitude_dividor.shape: ", filter_magnitude_dividor.shape)

                    cos_similarity = tf.divide(conv_reduce, tf.multiply(word_magnitude_dividor, filter_magnitude_dividor), name=v_name+"cos_similarity") #batch_size, seq_len - filter_len + 1, num_filter
                    print("filter_conv / cos_similarity.shape: ", cos_similarity.shape)
                    cos_similarity_expand = tf.expand_dims(cos_similarity, 2)

                    pool = tf.nn.max_pool(
                        cos_similarity_expand,
                        ksize=[1, sequence_length - filter_length + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID')
                    output = tf.reshape(pool, [-1, num_filter])
                    print("filter_conv / output.shape: ", output.shape)
                    
                    if cluster_list != None:
                        last_idx = 0
                        clustering_pooled = []
                        for idx in range(len(cluster_list)):
                            partial_pooled = tf.reduce_max(tf.slice(output, [0, last_idx], [-1, cluster_list[idx]]), axis=1)
                            clustering_pooled.append(tf.expand_dims(partial_pooled, 1))
                            last_idx += cluster_list[idx]
                        clustering_output = tf.concat(clustering_pooled, 1)
                        output = clustering_output
                        print("filter_conv / clustering_output.shape: ", clustering_output.shape)
                    return output
                
                single_outputs = []
                single_outputs.append(filter_conv(pos_list[0][0], pos_list[0][1], 1, "pos_1"))
                #single_outputs.append(filter_conv(pos_list[0], None, 1, "pos_1"))
                
                single_outputs.append(filter_conv(pos_list[1], None, 2, "pos_2"))
                single_outputs.append(filter_conv(pos_list[2], None, 3, "pos_3"))
                
                single_outputs.append(filter_conv(neg_list[0][0], neg_list[0][1], 1, "neg_1"))
                #single_outputs.append(filter_conv(neg_list[0], None, 1, "neg_1"))
                
                single_outputs.append(filter_conv(neg_list[1], None, 2, "neg_2"))
                single_outputs.append(filter_conv(neg_list[2], None, 3, "neg_3"))
                '''
                single_outputs.append(filter_conv(negation_list[0], None, 1, "negation_1"))
                single_outputs.append(filter_conv(negation_list[1], None, 2, "negation_2"))
                single_outputs.append(filter_conv(intensifier_list[0], None, 1, "intensifier_1"))
                single_outputs.append(filter_conv(intensifier_list[1], None, 2, "intensifier_2"))
                single_outputs.append(filter_conv(intensifier_list[2], None, 3, "intensifier_3"))
                '''
                self.knowledge_channel_output = tf.concat(single_outputs, 1)
                print("knowledge_channel_output.shape: ", self.knowledge_channel_output.shape)
                self.knowledge_output_length = self.knowledge_channel_output.shape.as_list()[1]
                

        if channel_setting != 'knowledge':
            #general channel
            general_pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_size, 1, num_filters] 
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    #W = tf.Variable(tf.contrib.layers.xavier_initializer()((filter_size, embedding_size, 1, num_filters)))
                    #print("Using xavier_initializer...")
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    conv = tf.nn.conv2d(
                        self.word_embedding_non_static_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    print("general channel shape of conv: ", conv.shape)
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, sequence_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    print("general channel shape of pooled: ", pooled.shape)
                    general_pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            self.h_pool = tf.concat(general_pooled_outputs, 3)
            self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])
            print("general channel shape of h_pool_flat: ", self.h_pool_flat.shape)

        if channel_setting == 'both':
            self.final_output = tf.concat([self.knowledge_channel_output, self.h_pool_flat], -1)
            self.final_output_length = num_filters_total + self.knowledge_output_length
        elif channel_setting == 'knowledge':
            self.final_output = self.knowledge_channel_output
            self.final_output_length = self.knowledge_output_length
        elif channel_setting == 'general':
            self.final_output = self.h_pool_flat
            self.final_output_length = num_filters_total
        print("shape of final_output: ", self.final_output.shape)

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.final_output, self.dropout_keep_prob)

        #with tf.name_scope("fully_connected"):
            #dense1 = tf.layers.dense(inputs=self.h_drop, units=128, activation=tf.nn.relu, name="dense1")
            #dense2 = tf.layers.dense(inputs=dense1, units=32, activation=tf.nn.relu, name="dense2")
            #dense3 = tf.layers.dense(inputs=self.h_drop, units=num_classes, name="dense3")

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            
            output_W = tf.get_variable(
                "output_W",
                shape=[self.final_output_length, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="output_b")

            l2_loss += tf.nn.l2_loss(output_W)
            l2_loss += tf.nn.l2_loss(output_b)

            self.scores = tf.nn.xw_plus_b(self.final_output, output_W, output_b, name="scores")
            
            #self.scores = dense3
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
