import tensorflow as tf

from model import Model
from preprocessing import Preprocessing

model = Model()
tf.reset_default_graph()

epochs = 100
batch_size = 64
rnn_len = 512
n_layers = 3
len_encoder_embeddings = 512
len_decoder_embeddings = 512
learning_rate = 0.01
learning_rate_decay = 0.9
min_learning_rate = 0.0001
prob_dropout = 0.5
inputs, outputs, lr, keep_prob = model.inputs_outputs()
session = tf.InteractiveSession()
len_seq = tf.placeholder_with_default(25, None, name='len_seq')
dim_inputs = tf.shape(inputs)

preprocessing = Preprocessing(dataset_name="movie-corpus")

orderned_qtn, orderned_ans = preprocessing.preprocessing()

prediction_train, prediction_test = model.model_seq2seq(
    tf.reverse(inputs, [-1]),
    outputs,
    keep_prob,
    len_seq,
    len(orderned_ans),
    len(orderned_qtn),
    len_encoder_embeddings,
    len_decoder_embeddings,
    rnn_len,
    n_layers,
    orderned_qtn
)

with tf.name_scope("optimization"):
    error = tf.contrib.seq2seq.sequence_loss(
        prediction_train, outputs, tf.ones([dim_inputs[0], len_seq]))
    optimizer = tf.train.AdamOptimizer(learning_rate)

    gradient = optimizer.compute_gradients(error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0, 5.0), grad_variable)
                         for grad_tensor, grad_variable in gradient if grad_tensor is not None]
    optimizer_clipping = optimizer.apply_gradients(clipped_gradients)
