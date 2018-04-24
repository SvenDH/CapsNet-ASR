import tensorflow as tf
import numpy as np

epsilon = 1e-9
batch_size = 64
iter_routing = 3
mask_with_y = True
m_plus = 0.9
m_minus = 0.1
lambda_val = 0.5
regularization_scale = 0.392
stddev = 0.01

class CapsLayer(object):
    ''' Capsule layer.
    Args:
        input: A 4-D tensor.
        num_outputs: the number of capsule in this layer.
        vec_len: integer, the length of the output vector of a capsule.
        layer_type: string, one of 'FC' or "CONV", the type of this layer,
            fully connected or convolution, for the future expansion capability
        with_routing: boolean, this capsule is routing with the
                      lower-level layer capsule.
    Returns:
        A 4-D tensor.
    '''
    def __init__(self, num_outputs, vec_len, with_routing=True, layer_type='FC'):
        self.num_outputs = num_outputs
        self.vec_len = vec_len
        self.with_routing = with_routing
        self.layer_type = layer_type

    def __call__(self, input, kernel_size=None, stride=None):
        '''
        The parameters 'kernel_size' and 'stride' will be used while 'layer_type' equal 'CONV'
        '''
        if self.layer_type == 'CONV':
            self.kernel_size = kernel_size
            self.stride = stride

            if not self.with_routing:
                # the PrimaryCaps layer, a convolutional layer
                # input: [batch_size, 20, 20, 256]
                capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                                                    self.kernel_size, self.stride, padding="VALID",
                                                    activation_fn=tf.nn.relu)
                # capsules = tf.contrib.layers.conv2d(input, self.num_outputs * self.vec_len,
                #                                    self.kernel_size, self.stride,padding="VALID",
                #                                    activation_fn=None)
                capsules = tf.reshape(capsules, (batch_size, -1, self.vec_len, 1))

                # [batch_size, 1152, 8, 1]
                capsules = squash(capsules)
                return(capsules)

        if self.layer_type == 'FC':
            if self.with_routing:
                # the DigitCaps layer, a fully connected layer
                # Reshape the input into [batch_size, 1152, 1, 8, 1]
                self.input = tf.reshape(input, shape=(batch_size, -1, 1, input.shape[-2].value, 1))

                with tf.variable_scope('routing'):
                    # b_IJ: [batch_size, num_caps_l, num_caps_l_plus_1, 1, 1],
                    # about the reason of using 'batch_size', see issue #21
                    b_IJ = tf.constant(np.zeros([batch_size, input.shape[1].value, self.num_outputs, 1, 1], dtype=np.float32))
                    capsules = routing(self.input, b_IJ)
                    capsules = tf.squeeze(capsules, axis=1)

            return(capsules)


def routing(input, b_IJ):
    ''' The routing algorithm.
    Args:
        input: A Tensor with [batch_size, num_caps_l=1152, 1, length(u_i)=8, 1]
               shape, num_caps_l meaning the number of capsule in the layer l.
    Returns:
        A Tensor of shape [batch_size, num_caps_l_plus_1, length(v_j)=16, 1]
        representing the vector output `v_j` in the layer l+1
    Notes:
        u_i represents the vector output of capsule i in the layer l, and
        v_j the vector output of capsule j in the layer l+1.
     '''

    # W: [1, num_caps_i, num_caps_j * len_v_j, len_u_j, 1]
    W = tf.get_variable('Weight', shape=(1, 8000, 976, 8, 1), dtype=tf.float32,
                        initializer=tf.random_normal_initializer(stddev=stddev))
    biases = tf.get_variable('bias', shape=(1, 1, 61, 16, 1))

    # Eq.2, calc u_hat
    # Since tf.matmul is a time-consuming op,
    # A better solution is using element-wise multiply, reduce_sum and reshape
    # ops instead. Matmul [a, b] x [b, c] is equal to a series ops as
    # element-wise multiply [a*c, b] * [a*c, b], reduce_sum at axis=1 and
    # reshape to [a, c]
    input = tf.tile(input, [1, 1, 976, 1, 1])

    u_hat = tf.reduce_sum(W * input, axis=3, keepdims=True)
    u_hat = tf.reshape(u_hat, shape=[-1, 8000, 61, 16, 1])

    # In forward, u_hat_stopped = u_hat; in backward, no gradient passed back from u_hat_stopped to u_hat
    u_hat_stopped = tf.stop_gradient(u_hat, name='stop_gradient')

    # line 3,for r iterations do
    for r_iter in range(iter_routing):
        with tf.variable_scope('iter_' + str(r_iter)):
            # line 4:
            # => [batch_size, 1152, 10, 1, 1]
            c_IJ = tf.nn.softmax(b_IJ, axis=2)

            # At last iteration, use `u_hat` in order to receive gradients from the following graph
            if r_iter == iter_routing - 1:
                # line 5:
                # weighting u_hat with c_IJ, element-wise in the last two dims
                # => [batch_size, 1152, 10, 16, 1]
                s_J = tf.multiply(c_IJ, u_hat)
                # then sum in the second dim, resulting in [batch_size, 1, 10, 16, 1]
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases

                # line 6:
                # squash using Eq.1,
                v_J = squash(s_J)
            elif r_iter < iter_routing - 1:  # Inner iterations, do not apply backpropagation
                s_J = tf.multiply(c_IJ, u_hat_stopped)
                s_J = tf.reduce_sum(s_J, axis=1, keepdims=True) + biases
                v_J = squash(s_J)

                # line 7:
                # reshape & tile v_j from [batch_size ,1, 10, 16, 1] to [batch_size, 1152, 10, 16, 1]
                # then matmul in the last tow dim: [16, 1].T x [16, 1] => [1, 1], reduce mean in the
                # batch_size dim, resulting in [1, 1152, 10, 1, 1]
                v_J_tiled = tf.tile(v_J, [1, 8000, 1, 1, 1])
                u_produce_v = tf.reduce_sum(u_hat_stopped * v_J_tiled, axis=3, keepdims=True)

                # b_IJ += tf.reduce_sum(u_produce_v, axis=0, keep_dims=True)
                b_IJ += u_produce_v

    return(v_J)


def squash(vector):
    '''Squashing function corresponding to Eq. 1
    Args:
        vector: A tensor with shape [batch_size, 1, num_caps, vec_len, 1] or [batch_size, num_caps, vec_len, 1].
    Returns:
        A tensor with the same shape as vector but squashed in 'vec_len' dimension.
    '''
    vec_squared_norm = tf.reduce_sum(tf.square(vector), -2, keepdims=True)
    scalar_factor = vec_squared_norm / (1 + vec_squared_norm) / tf.sqrt(vec_squared_norm + epsilon)
    vec_squashed = scalar_factor * vector  # element-wise
    return(vec_squashed)


class CapsNet(object):
    def __init__(self, data, labels, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.X = data
                self.labels = labels
                self.Y = tf.one_hot(self.labels, depth=61, axis=1, dtype=tf.float32)

                self.build_arch()
                self.loss()
                self._summary()

                # t_vars = tf.trainable_variables()
                self.global_step = tf.Variable(0, name='global_step', trainable=False)
                self.optimizer = tf.train.AdamOptimizer()
                self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)  # var_list=t_vars)
            else:
                self.X = tf.placeholder(tf.float32, shape=(batch_size, 7, 256, 1))
                self.labels = tf.placeholder(tf.int32, shape=(batch_size, ))
                self.Y = tf.reshape(self.labels, shape=(batch_size, 61, 1))
                self.build_arch()

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 5, 252, 256]
            conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=[3,5], stride=1,
                                             padding='VALID')

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            primaryCaps = CapsLayer(num_outputs=32, vec_len=8, with_routing=False, layer_type='CONV')
            caps1 = primaryCaps(conv1, kernel_size=3, stride=2)

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            digitCaps = CapsLayer(num_outputs=61, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = digitCaps(caps1)

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2),
                                               axis=2, keepdims=True) + epsilon)
            self.softmax_v = tf.nn.softmax(self.v_length, axis=1)

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(batch_size, ))

            # Method 1.
            if not mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx
                # as we are 3-dim animal
                masked_v = []
                for i in range(batch_size):
                    v = self.caps2[i][self.argmax_idx[i], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
            # Method 2. masking with true label, default mode
            else:
                # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
                self.v_length = tf.sqrt(tf.reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=784, activation_fn=tf.sigmoid)

    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - m_minus))

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(batch_size, -1))
        max_r = tf.reshape(max_r, shape=(batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.total_loss = self.margin_loss + regularization_scale * self.reconstruction_err

    # Summary
    def _summary(self):
        train_summary = []
        train_summary.append(tf.summary.scalar('train/margin_loss', self.margin_loss))
        train_summary.append(tf.summary.scalar('train/reconstruction_loss', self.reconstruction_err))
        train_summary.append(tf.summary.scalar('train/total_loss', self.total_loss))
        recon_img = tf.reshape(self.decoded, shape=(batch_size, 28, 28, 1))
        train_summary.append(tf.summary.image('reconstruction_img', recon_img))
        self.train_summary = tf.summary.merge(train_summary)

        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))