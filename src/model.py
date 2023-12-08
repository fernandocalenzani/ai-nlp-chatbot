import tensorflow as tf


class Model:
    """     def __init__(self, preprocessing):
            #self.preprocessing = preprocessing
    """

    def inputs_outputs(self):
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        outputs = tf.placeholder(tf.int32, [None, None], name='outputs')
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        return inputs, outputs, lr, keep_prob

    def preprocessing_outputs(self, outputs, data, batch_size):
        left = tf.fill([batch_size, 1], data['<SOS>'])
        right = tf.strided_slice(
            outputs, [0, 0], [batch_size, -1], strides=[1, 1])
        outputs_preprocessed = tf.concat([left, right], 1)

        return outputs_preprocessed

    def rnn_encoder(self, rnn_inputs, rnn_len, n_layers, keep_prob, len_seq):
        lstm = tf.contrib.rnn.LSTMCell(rnn_len)
        lstm_out = tf.contrib.rnn.DropoutWrapper(
            lstm, input_keep_prob=keep_prob)
        encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_out] * n_layers)
        encoder_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=encoder_cell, cell_bw=encoder_cell, sequence_length=len_seq, inputs=rnn_inputs, dtype=tf.float32)

        return encoder_state

    def decoder_base_train(self, encoder_state, decoder_cell, decoder_embedded_input, len_seq, decoder_scope, outputs, keep_prob, batch_size):

        states_attention = tf.zeros([batch_size, 1, decoder_cell.output_size])
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
            states_attention, attention_option='badanau', num_units=decoder_cell.output_size)

        function_decoder_train = tf.contrib.seq2seq.attention_decoder_fn_train(
            encoder_state[0],
            attention_keys,
            attention_values,
            attention_score_function,
            attention_construct_function,
            name='attention_decoder_train'
        )

        decoder_output, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            decoder_cell,
            function_decoder_train,
            decoder_embedded_input,
            len_seq,
            scope=decoder_scope
        )

        decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)

        return outputs(decoder_output_dropout)

    def decoder_base_test(self, encoder_state, decoder_cell, decoder_embedded_matrix, sos_id, eos_id, len_max, n_words, decoder_scope, outputs, keep_prob, batch_size):

        states_attention = tf.zeros(
            [batch_size, 1, decoder_cell.output_size])
        attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
            states_attention, attention_option='badanau', num_units=decoder_cell.output_size)

        function_decoder_test = tf.contrib.seq2seq.attention_decoder_fn_inference(
            outputs,
            encoder_state[0],
            attention_keys,
            attention_values,
            attention_score_function,
            attention_construct_function,
            decoder_embedded_matrix,
            sos_id,
            eos_id,
            len_max,
            n_words,
            name='attention_decoder_inf'
        )

        predictions_test, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            decoder_cell,
            function_decoder_test,
            scope=decoder_scope
        )

        return predictions_test

    def rnn_decoder(self, decoder_embedded_inputs, decoder_embedded_matrix, encoder_state, n_words, len_seq, rnn_len, n_layers, data, keep_prob, batch_size):

        with tf.variable_scope("decoder") as decoder_scope:
            lstm = tf.contrib.rnn.LSTMCell(rnn_len)
            lstm_out = tf.contrib.rnn.DropoutWrapper(
                lstm, input_keep_prob=keep_prob)
            decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_out] * n_layers)

            weight = tf.truncate_normal_initializer(stddev=0.1)
            biases = tf.zeros_initializer()

            def output(x): return tf.contrib.layers.fully_connected(
                x, n_words, None, scope=decoder_scope, weights_initializer=weight, biases_initializer=biases)

            prediction_train = self.decoder_base_train(
                self, encoder_state, decoder_cell, decoder_embedded_inputs, len_seq, decoder_scope, output, keep_prob, batch_size)

            decoder_scope.reuse_variable()

            prediction_test = self.decoder_base_test(encoder_state, decoder_cell, decoder_embedded_matrix,
                                                     data['<SOS>'], data['<EOS>'], len_seq - 1, n_words, decoder_scope, output, keep_prob, batch_size)

        return prediction_train, prediction_test

    def model_seq2seq(self, inputs, outputs, keep_prob, batch_size, len_seq, n_words_ans, n_words_qtn, len_encode_embedded, len_decode_embedded, rnn_len, n_layers, data):

        encode_embedded_input = tf.contrib.layers.embed_sequence(
            inputs, n_words_qtn+1, len_encode_embedded, initializer=tf.random.uniform_initializer(0, 1))

        encode_state = self.rnn_encoder(
            encode_embedded_input, rnn_len, n_layers, keep_prob, len_seq)

        output_processing = self.preprocessing_outputs(
            outputs, data, batch_size)
        decode_embedded_matrix = tf.Variable(
            tf.random_normal([n_words_qtn + 1, len_decode_embedded]))

        decode_embedded_input = tf.nn.embedding_lookup(
            decode_embedded_matrix, output_processing)

        predict_train, predict_test = self.rnn_decoder(
            decode_embedded_input, decode_embedded_matrix, encode_state, n_words_ans, len_seq, rnn_len, data, keep_prob, batch_size)

        return predict_train, predict_test
