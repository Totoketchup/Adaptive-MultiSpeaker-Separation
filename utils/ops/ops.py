import numpy as np
import tensorflow as tf
import functools


rng = np.random.RandomState(1234)

def scope(function):
    name = function.__name__
    attribute = '_cache_' + name
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self,attribute):
            with tf.variable_scope(name):
                setattr(self,attribute,function(self))
        return getattr(self,attribute)
    return decorator


def Xavier_init(in_dim, hid_dim, name):
    return tf.Variable(rng.uniform(
        low=-np.sqrt(6/(in_dim + hid_dim)),
        high=np.sqrt(6/(in_dim + hid_dim)),
        size=(in_dim, hid_dim)
    ).astype('float32'), name=name)


class Embedding:
    def __init__(self, vocab_size, emb_dim, scale=0.08):
        self.V = tf.Variable(rng.randn(vocab_size, emb_dim).astype('float32') * scale, name='V')

    def f_prop(self, x):
        return tf.nn.embedding_lookup(self.V, x)

    
class RNN:
    def __init__(self, in_dim, hid_dim, m, scale=0.08):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        # Xavier initializer
        self.W_in = Xavier_init(in_dim, hid_dim, name='W_in')
        # Random orthogonal initializer
        self.W_re = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_re')
        self.b_re = tf.Variable(tf.zeros([hid_dim], dtype=tf.float32), name='b_re')
        self.m = m

    def f_prop(self, x):
        def fn(h_tm1, x_and_m):
            x = x_and_m[0]
            m = x_and_m[1]
            
            
            h_t = tf.nn.tanh(tf.matmul(h_tm1, self.W_re) + tf.matmul(x, self.W_in) + self.b_re)
            return m[:, None] * h_t + (1 - m[:, None]) * h_tm1 # Mask

        # shape: [batch_size, sentence_length, in_dim] -> shape: [sentence_length, batch_size, in_dim]
        _x = tf.transpose(x, perm=[1, 0, 2])
        # shape: [batch_size, sentence_length] -> shape: [sentence_length, batch_size]
        _m = tf.transpose(self.m)
        h_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim])) # Initial state

        h = tf.scan(fn=fn, elems=[_x, _m], initializer=h_0)

        return h[-1] # Take the last state
 

class LSTM:
    def __init__(self, in_dim, hid_dim, m, scale=0.08):
        self.in_dim = in_dim
        self.hid_dim = hid_dim

        # Xavier initializer
        self.W_fin = Xavier_Variable(in_dim, hid_dim, 'W_fin')
        self.W_iin = Xavier_Variable(in_dim, hid_dim, 'W_iin')
        self.W_cin = Xavier_Variable(in_dim, hid_dim, 'W_cin')
        self.W_oin = Xavier_Variable(in_dim, hid_dim, 'W_oin')

        # Random orthogonal initializer
        self.W_f = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_f')
        self.b_f = tf.Variable(tf.zeros([hid_dim], dtype=tf.float32), name='b_f')
        self.W_i = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_i')
        self.b_i = tf.Variable(tf.zeros([hid_dim], dtype=tf.float32), name='b_i')
        self.W_c = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_c')
        self.b_c = tf.Variable(tf.zeros([hid_dim], dtype=tf.float32), name='b_c')
        self.W_o = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_o')
        self.b_o = tf.Variable(tf.zeros([hid_dim], dtype=tf.float32), name='b_o')

        self.m = m

    def f_prop(self, x):
        def fn(c_h_tm1, x_and_m):
            x = x_and_m[0]
            m = x_and_m[1]
            
            c_tm1 = c_h_tm1[0] # Cell state ct-1
            h_tm1 = c_h_tm1[1] # Output state ht-1
            
            g_f = tf.nn.sigmoid(tf.matmul(h_tm1, self.W_f) + tf.matmul(x, self.W_fin) + self.b_f)
            i = tf.nn.sigmoid(tf.matmul(h_tm1, self.W_i) + tf.matmul(x, self.W_iin) + self.b_i)
            g_o = tf.nn.sigmoid(tf.matmul(h_tm1, self.W_o) + tf.matmul(x, self.W_oin) + self.b_o)
            g_c = tf.nn.tanh(tf.matmul(h_tm1, self.W_c) + tf.matmul(x, self.W_cin) + self.b_c)
            
            c_t = c_tm1 * g_f + i * g_c
            h_t = tf.nn.tanh(c_t) * g_o
            
            h_t = m[:, None] * h_t + (1 - m[:, None]) * h_tm1 # Mask
            return [c_t, h_t]

        # shape: [batch_size, sentence_length, in_dim] -> shape: [sentence_length, batch_size, in_dim]
        _x = tf.transpose(x, perm=[1, 0, 2])
        # shape: [batch_size, sentence_length] -> shape: [sentence_length, batch_size]
        _m = tf.transpose(self.m)
        h_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim])) # Initial state
        c_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim])) # Initial state

        h = tf.scan(fn=fn, elems=[_x, _m], initializer=[c_0, h_0])
        return h[1][-1] # Take the last state
        
class Dense:
    def __init__(self, in_dim, out_dim, function=lambda x: x, name='Dense'):
        # Xavier initializer
        self.W = Xavier_init(in_dim, out_dim, 'W')
        self.b = tf.Variable(np.zeros([out_dim]).astype('float32'))
        self.function = function
        self.name = name


    def f_prop(self, x):
        return self.function(tf.matmul(x, self.W) + self.b)


class Pooling:
    def __init__(self, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME'):
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        return tf.nn.max_pool(x, ksize=self.ksize, strides=self.strides, padding=self.padding)

class Global_Average_Pooling:
    def __init__(self):
        pass

    def f_prop(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

class Flatten:
    def f_prop(self, x):
        return tf.reshape(x, (-1, np.prod(x.get_shape().as_list()[1:])))


class Activation:
    def __init__(self, function=lambda x: x):
        self.function = function

    def f_prop(self, x):
        return self.function(x)
    
class Dropout:
    def __init__(self, prob, train):
        self.prob = prob
        self.train = train

    def f_prop(self, x):
        return tf.cond(self.train, lambda: tf.nn.dropout(x, self.prob), lambda: x)

class Reshape:
    def __init__(self, shape, name = 'Reshape'):
        self.shape = shape
        self.name = name

    def f_prop(self, x):
        return tf.reshape(x, self.shape)

class Normalize:
    def __init__(self, axis, name = 'Normalize'):
        self.axis = axis
        self.name = name

    def f_prop(self, x):
        return tf.nn.l2_normalize(x, self.axis)


class BatchNorm:
    def __init__(self, shape, epsilon=np.float32(1e-5)):
        self.gamma = tf.Variable(np.ones(shape, dtype='float32'), name='gamma')
        self.beta  = tf.Variable(np.zeros(shape, dtype='float32'), name='beta')
        self.epsilon = epsilon

    def f_prop(self, x):
        if len(x.get_shape()) == 2:
            mean, var = tf.nn.moments(x, axes=0, keepdims=True)
            std = tf.sqrt(var + self.epsilon)
        elif len(x.get_shape()) == 4:
            mean, var = tf.nn.moments(x, axes=(0,1,2), keep_dims=True)
            std = tf.sqrt(var + self.epsilon)
        normalized_x = (x - mean) / std
        return self.gamma * normalized_x + self.beta

class BLSTM:
    def __init__(self, hid_dim, name):
        self.hid_dim = hid_dim
        self.name = name

    def f_prop(self, x):
        forward_input = x
        backward_input = tf.reverse(x, [1])

        # Forward pass
        with tf.variable_scope('forward_' + self.name):
            forward_lstm = tf.contrib.rnn.BasicLSTMCell(self.hid_dim//2)
            forward_out, _ = tf.nn.dynamic_rnn(forward_lstm, forward_input, dtype=tf.float32)

        # backaward pass
        with tf.variable_scope('backward_' + self.name):
            backward_lstm = tf.contrib.rnn.BasicLSTMCell(self.hid_dim//2)
            backward_out, _ = tf.nn.dynamic_rnn(backward_lstm, backward_input, dtype=tf.float32)

        # Concatenate the RNN outputs and return
        return tf.concat([forward_out[:,:,:], backward_out[:,::-1,:]], 2)
        
class Residual:
    def __init__(self, layers):
        self.layers = layers;
        self.downsample = False;
        if layers[0].d_in != layers[0].d_out:
            self.downsample = True;
            self.conv_op = Conv((1,1,layers[0].d_in,layers[0].d_out), strides = layers[0].strides)
            
    def f_prop(self, x):
        o = x;
        for layer in self.layers:
            o = layer.f_prop(o)

        if self.downsample:
            x =  self.conv_op.f_prop(x) # Add the input to the residual learning 
        return o + x;

class Residual_Net:
    def __init__(self, input_shape, training, k = [1, 32, 64, 128], N = 3, name='Residual Net'):
        self.k = k
        self.T = input_shape[0]
        self.F = input_shape[1]
        self.name = name
        blocks = []
        for i in range(len(k)-1):
            strides = [1, 2, 2, 1]
            T = int(self.T/pow(2,i))
            F = int(self.F/pow(2,i))# + self.F%(self.F/pow(2,i))) 
            
            for n in range(N):
                if i == 0 or n != 0:
                    strides = [1, 1, 1, 1]
                blocks.append([
                    Conv((3, 3, k[i+int(n!=0)], k[i+1]), strides=strides), # TxFx1 -> TxFx32
                    BatchNorm((T, F, k[i+1])),
                    Activation(tf.nn.relu),
                    Dropout(0.7, training),
                    Conv((3, 3,  k[i+1],  k[i+1])), # TxFx32 -> TxFx32
                    BatchNorm((T, F,  k[i+1])),
                    Activation(tf.nn.relu),
                ])
        self.layers = []

        # Create the Residual blocks
        for b in blocks:
            self.layers.append(Residual(b))
            
    def f_prop(self, x):
        for layer in self.layers:
            x = layer.f_prop(x)
        return x

 
##############################
### CONVOLUTION OPERATIONS ###
##############################

class Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], padding='SAME'):
        # Xavier
        self.filter = filter_shape
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.d_in = filter_shape[2]
        self.d_out = filter_shape[3]
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(6/(fan_in + fan_out)),
                        high=np.sqrt(6/(fan_in + fan_out)),
                        size=filter_shape
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b')
        self.function = function
        self.strides = strides
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.conv2d(x, self.W, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)


class Conv1D:
    def __init__(self, filter_shape, function=lambda x: x, stride=1, padding='SAME', name='Conv1D'):
        # Xavier
        fan_in = np.sqrt(2/(float(filter_shape[1]+filter_shape[2])))
        self.W = tf.Variable(rng.uniform(
                        low=-np.sqrt(2/fan_in),
                        high=np.sqrt(2/fan_in),
                        size=filter_shape
                    ).astype('float32'), name='W')
        self.b = tf.Variable(np.zeros((filter_shape[-1]), dtype='float32'), name='b')
        self.function = function
        self.stride = stride
        self.padding = padding
        self.name = name

    def f_prop(self, x):
        u = tf.nn.conv1d(x, self.W, stride=self.stride, padding=self.padding)
        return self.function(u + self.b)

class DeConv:
    def __init__(self, filter_shape, output_shape, function=lambda x: x, strides=[1,1,1,1], padding='SAME'):
        # Xavier
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.d_in = filter_shape[2]
        self.d_out = filter_shape[3]
        self.W = Xavier_init(fan_in, fan_out, name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b')
        self.function = function
        self.strides = strides
        self.output_shape = output_shape
        self.rate = rate
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.tf.nn.conv2d_transpose(x, self.W, output_shape=self.output_shape, strides=self.strides, padding=self.padding) + self.b
        return self.function(u)
 

class Dilated_Conv:
    def __init__(self, filter_shape, function=lambda x: x, strides=[1,1,1,1], rate=2, padding='SAME'):
        # Xavier
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.d_in = filter_shape[2]
        self.d_out = filter_shape[3]
        self.W = Xavier_init(fan_in, fan_out, name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b')
        self.function = function
        self.strides = strides
        self.rate = rate
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.atrous_conv2d(x, self.W, strides=self.strides, rate=self.rate, padding=self.padding) + self.b
        return self.function(u)
 
class Dilated_DeConv:
    def __init__(self, filter_shape, output_shape, function=lambda x: x, strides=[1,1,1,1], rate=2, padding='SAME'):
        # Xavier
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.d_in = filter_shape[2]
        self.d_out = filter_shape[3]
        self.W = Xavier_init(fan_in, fan_out, name='W')
        self.b = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b')
        self.function = function
        self.strides = strides
        self.output_shape = output_shape
        self.rate = rate
        self.padding = padding

    def f_prop(self, x):
        u = tf.nn.atrous_conv2d_transpose(x, self.W, output_shape=self.output_shape, strides=self.strides, rate=self.rate, padding=self.padding) + self.b
        return self.function(u)

# Based on: Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting

class Conv_LSTM:
    def __init__(self, filter_shape, in_dim, hid_dim, m, scale=0.08):
        self.in_dim = in_dim
        self.hid_dim = hid_dim
        
        # Conv init
        fan_in = np.prod(filter_shape[:3])
        fan_out = np.prod(filter_shape[:2]) * filter_shape[3]
        self.d_in = filter_shape[2]
        self.d_out = filter_shape[3]
        self.W = Xavier_init(fan_in, fan_out, name='W')


        # Xavier initializer
        self.W_fin = Xavier_Variable(fan_in, fan_out, 'W_fin')
        self.W_iin = Xavier_Variable(fan_in, fan_out, 'W_iin')
        self.W_cin = Xavier_Variable(fan_in, fan_out, 'W_cin')
        self.W_oin = Xavier_Variable(fan_in, fan_out, 'W_oin')

        # Random orthogonal initializer
        self.W_f = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_f')
        self.b_f = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b_f')
        self.W_i = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_i')
        self.b_i = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b_i')
        self.W_c = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_c')
        self.b_c = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b_c')
        self.W_o = tf.Variable(orthogonal_initializer((hid_dim, hid_dim)), name='W_o')
        self.b_o = tf.Variable(np.zeros((filter_shape[3]), dtype='float32'), name='b_o')

        self.m = m

        #Convolutional Operation
        self.op = lambda x, W: tf.nn.conv2d(x, W, strides=self.strides, padding=self.padding)

    def f_prop(self, x):
        def fn(c_h_tm1, x_and_m):
            x = x_and_m[0]
            m = x_and_m[1]
            
            c_tm1 = c_h_tm1[0] # Cell state ct-1
            h_tm1 = c_h_tm1[1] # Output state ht-1
            
            g_f = tf.nn.sigmoid(op(h_tm1, self.W_f) + op(x, self.W_fin) + self.b_f)
            i = tf.nn.sigmoid(op(h_tm1, self.W_i) + op(x, self.W_iin) + self.b_i)
            g_o = tf.nn.sigmoid(op(h_tm1, self.W_o) + op(x, self.W_oin) + self.b_o)
            g_c = tf.nn.tanh(op(h_tm1, self.W_c) + op(x, self.W_cin) + self.b_c)
            
            c_t = c_tm1 * g_f + i * g_c
            h_t = tf.nn.tanh(c_t) * g_o
            
            h_t = m[:, None] * h_t + (1 - m[:, None]) * h_tm1 # Mask
            return [c_t, h_t]

        # shape: [batch_size, sentence_length, in_dim] -> shape: [sentence_length, batch_size, in_dim]
        _x = tf.transpose(x, perm=[1, 0, 2])
        # shape: [batch_size, sentence_length] -> shape: [sentence_length, batch_size]
        _m = tf.transpose(self.m)
        h_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim])) # Initial state
        c_0 = tf.matmul(x[:, 0, :], tf.zeros([self.in_dim, self.hid_dim])) # Initial state

        h = tf.scan(fn=fn, elems=[_x, _m], initializer=[c_0, h_0])
        return h[1] # Output