import numpy as np
import tensorflow as tf
import functools

rng = np.random.RandomState(42)


def normalize_mix(X_mix, X_non_mix, type_='min-max'):
    if type_ == 'min-max':
        a = -1.0
        b = 1.0
        max_val = np.amax(X_mix, axis=-1, keepdims=True)
        min_val = np.amin(X_mix, axis=-1, keepdims=True)

        S = float(X_non_mix.shape[1])
        A = (b - a)/(max_val - min_val)
        B = b - A * max_val
        X_mix = A*X_mix + B

        X_non_mix = A[:,:,np.newaxis]*X_non_mix + B[:,:,np.newaxis]/S
        val1 = min_val
        val2 = max_val
    elif type_ == 'mean-std':
        mean = np.mean(X_mix, axis=-1, keepdims=True)
        std = np.std(X_mix, axis=-1, keepdims=True)
        S = float(X_non_mix.shape[1])
        X_mix = (X_mix - mean)
        X_non_mix = X_non_mix - mean/(S)
        val1 = mean
        val2 = std

    return X_mix, X_non_mix, val1, val2

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

def logfunc(x, x2):
    cx = tf.clip_by_value(x, 1e-10, 1.0)
    cx2 = tf.clip_by_value(x2, 1e-10, 1.0)
    return tf.multiply(x, tf.log(tf.div(cx,cx2)))

def kl_div(p, p_hat):
    inv_p = 1 - p
    inv_p_hat = 1 - p_hat 
    return logfunc(p, p_hat) + logfunc(inv_p, inv_p_hat)

def log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10., dtype=numerator.dtype))
    return numerator / denominator

def get_scope_variable(scope_name, var, shape=None, initializer=None):
    with tf.variable_scope(scope_name) as scope:
        try:
            v = tf.get_variable(var, shape, initializer=initializer)
        except ValueError:
            scope.reuse_variables()
            v = tf.get_variable(var)
    return v

def variable_summaries(var):
    # Attach a lot of summaries to a Tensor (for TensorBoard visualization)
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        # tf.summary.histogram('histogram', var)

def f_props(layers, x):
    for i, layer in enumerate(layers):
        x = layer.f_prop(x)
    return x

def Xavier_init(in_dim, hid_dim, name):
    return tf.Variable(rng.uniform(
        low=-np.sqrt(6/(in_dim + hid_dim)),
        high=np.sqrt(6/(in_dim + hid_dim)),
        size=(in_dim, hid_dim)
    ).astype('float32'), name=name)

def unpool(pool, ind, output_shape, ksize=[1, 2, 2, 1], scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
        input_shape =  tf.shape(pool)
        
        flat_input_size = tf.cumprod(input_shape)[-1]
        flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

        pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
        batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                          shape=tf.stack([input_shape[0], 1, 1, 1]))
        b = tf.ones_like(ind) * batch_range
        b = tf.reshape(b, tf.stack([flat_input_size, 1]))
        ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
        ind_ = tf.concat([b, ind_], 1)

        ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
        ret = tf.reshape(ret, tf.stack(output_shape))
        return ret

# def unpool(pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
#     """
#        Unpooling layer after max_pool_with_argmax.
#        Args:
#            pool:   max pooled output tensor
#            ind:      argmax indices
#            ksize:     ksize is the same as for the pool
#        Return:
#            unpool:    unpooling tensor
#     """
#     with tf.variable_scope(scope):
#         input_shape = tf.shape(pool)
#         output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

#         flat_input_size = tf.reduce_prod(input_shape)
#         flat_output_shape = [output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]]

#         pool_ = tf.reshape(pool, [flat_input_size])
#         batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
#                                           shape=[input_shape[0], 1, 1, 1])
#         b = tf.ones_like(ind) * batch_range
#         b1 = tf.reshape(b, [flat_input_size, 1])
#         ind_ = tf.reshape(ind, [flat_input_size, 1])
#         ind_ = tf.concat([b1, ind_], 1)

#         ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
#         ret = tf.reshape(ret, output_shape)

#         set_input_shape = pool.get_shape()
#         set_output_shape = [set_input_shape[0], set_input_shape[1] * ksize[1], set_input_shape[2] * ksize[2], set_input_shape[3]]
#         ret.set_shape(set_output_shape)
#         return ret
    # with tf.variable_scope(scope):
    #     pooled_shape = tf.shape(pool)

    #     flatten_ind = tf.reshape(ind, (pooled_shape[0], pooled_shape[1] * pooled_shape[2] * pooled_shape[3]))
    #     # sparse indices to dense ones_like matrics
    #     one_hot_ind = tf.one_hot(flatten_ind,  pooled_shape[1] * ksize[1] * pooled_shape[2] * ksize[2] * pooled_shape[3], on_value=1., off_value=0., axis=-1)
    #     one_hot_ind = tf.reduce_sum(one_hot_ind, axis=1)
    #     one_like_mask = tf.reshape(one_hot_ind, (pooled_shape[0], pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2], pooled_shape[3]))
    #     # resize input array to the output size by nearest neighbor
    #     img = tf.image.resize_nearest_neighbor(pool, [pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2]])
    #     unpooled = tf.multiply(img, tf.cast(one_like_mask, img.dtype))
    #     return unpooled


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
        # return tf.nn.dropout(x, self.prob, name='dropout_res')

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

class Abs:
    def __init__(self, axis, name = 'Abs'):
        self.name = name

    def f_prop(self, x):
        return tf.abs(x)


# class BatchNorm:
#     def __init__(self, shape, epsilon=np.float32(1e-5)):
#         self.gamma = tf.Variable(tf.ones(shape, dtype='float32'), validate_shape=False, name='gamma')
#         self.beta  = tf.Variable(tf.zeros(shape, dtype='float32'), validate_shape=False, name='beta')
#         self.epsilon = epsilon

#     def f_prop(self, x):
#         if len(x.get_shape()) == 2:
#             mean, var = tf.nn.moments(x, axes=0, keepdims=True)
#             std = tf.sqrt(var + self.epsilon)
#         elif len(x.get_shape()) == 4:
#             mean, var = tf.nn.moments(x, axes=(0,1,2), keep_dims=True)
#             std = tf.sqrt(var + self.epsilon)
#         normalized_x = (x - mean) / std
#         print normalized_x
#         return self.gamma * normalized_x + self.beta

class BatchNorm:
    def __init__(self, shape, epsilon=np.float32(1e-5)):
        self.shape = shape

    def f_prop(self, x):
        return tf.layers.batch_normalization(x)

class BLSTM:
    def __init__(self, hid_dim, name, dropout=False, drop_val=0.8):
        self.hid_dim = hid_dim
        self.name = name
        self.dropout = dropout
        self.drop_val = drop_val

    def f_prop(self, x):
        forward_input = x
        backward_input = tf.reverse(x, [1])

        # Forward pass
        with tf.variable_scope('forward_' + self.name):
            forward_lstm = tf.contrib.rnn.BasicLSTMCell(self.hid_dim//2)
            if self.dropout:
                forward_lstm = tf.contrib.rnn.DropoutWrapper(forward_lstm, self.drop_val, self.drop_val, self.drop_val)
            forward_out, _ = tf.nn.dynamic_rnn(forward_lstm, forward_input, dtype=tf.float32)

        # backaward pass
        with tf.variable_scope('backward_' + self.name):
            backward_lstm = tf.contrib.rnn.BasicLSTMCell(self.hid_dim//2)
            if self.dropout:
                backward_lstm = tf.contrib.rnn.DropoutWrapper(backward_lstm, self.drop_val, self.drop_val, self.drop_val)
            backward_out, _ = tf.nn.dynamic_rnn(backward_lstm, backward_input, dtype=tf.float32)

        # Concatenate the RNN outputs and return
        return tf.concat([forward_out[:,:,:], backward_out[:,::-1,:]], 2)

class Residual:
    def __init__(self, layers):
        self.layers = layers;
        self.downsample = False;
        if layers[0].strides[0] != 1: #layers[0].d_in != layers[0].d_out:
            self.downsample = True;
            self.conv_op = Conv2D(layers[0].filters, (1,1), layers[0].strides)
            
    def f_prop(self, x):
        o = x;
        for layer in self.layers:
            print layer
            o = layer.f_prop(o)

        if self.downsample:
            x =  self.conv_op # Add the input to the residual learning 

        if(tf.shape(x)[-1] != tf.shape(o)[-1]):
            x = Conv2D(self.layers[0].filters, (1,1), self.layers[0].strides).f_prop(x)

        return o + x;

class Residual_Net:
    def __init__(self, input_shape, training, k = [1, 32, 64, 128], N = 3, name='Residual Net'):
        self.k = k
        self.T = input_shape[0]
        self.F = input_shape[1]
        self.name = name
        blocks = []
        filter_shape = (3,3)
        for i in range(len(k)-1):
            # strides = (2,2)
            strides = (1,1)
            T = tf.cast(self.T/pow(2,i), tf.int32)
            F = tf.cast(self.F/pow(2,i) + self.F%(self.F/pow(2,i)), tf.int32)
            for n in range(N):
                if i == 0 or n != 0:
                    strides = (1,1)
                blocks.append([
                    Conv2D(k[i+1], filter_shape, strides),
                    BatchNorm((T, F, k[i+1])),
                    Activation(tf.nn.relu),
                    Dropout(0.7, training),
                    Conv2D(k[i+1], filter_shape, (1,1)),
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
        print x
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

class Conv2D:                             
    def __init__(self, filters, kernel, strides=(1,1), padding='same',
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        name='conv2d'):
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.name = name
        self.kernel_initializer = kernel_initializer
        
    def f_prop(self, x):
        return tf.layers.conv2d(x, self.filters, kernel_size=self.kernel, strides=self.strides,padding= self.padding,
        kernel_initializer=self.kernel_initializer)

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
            
            g_f = tf.nn.sigmoid(self.op(h_tm1, self.W_f) + self.op(x, self.W_fin) + self.b_f)
            i = tf.nn.sigmoid(self.op(h_tm1, self.W_i) + self.op(x, self.W_iin) + self.b_i)
            g_o = tf.nn.sigmoid(self.op(h_tm1, self.W_o) + self.op(x, self.W_oin) + self.b_o)
            g_c = tf.nn.tanh(self.op(h_tm1, self.W_c) + self.op(x, self.W_cin) + self.b_c)
            
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


"""AMSGrad for TensorFlow."""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer

class AMSGrad(optimizer.Optimizer):
    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.99, epsilon=1e-8, use_locking=False, name="AMSGrad"):
        super(AMSGrad, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._epsilon = epsilon

        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None
        self._epsilon_t = None

        self._beta1_power = None
        self._beta2_power = None

    def _create_slots(self, var_list):
        first_var = min(var_list, key=lambda x: x.name)

        create_new = self._beta1_power is None
        if not create_new and context.in_graph_mode():
            create_new = (self._beta1_power.graph is not first_var.graph)

        if create_new:
            with ops.colocate_with(first_var):
                self._beta1_power = variable_scope.variable(self._beta1, name="beta1_power", trainable=False)
                self._beta2_power = variable_scope.variable(self._beta2, name="beta2_power", trainable=False)
        # Create slots for the first and second moments.
        for v in var_list :
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)
            self._zeros_slot(v, "vhat", self._name)

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr)
        self._beta1_t = ops.convert_to_tensor(self._beta1)
        self._beta2_t = ops.convert_to_tensor(self._beta2)
        self._epsilon_t = ops.convert_to_tensor(self._epsilon)

    def _apply_dense(self, grad, var):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _resource_apply_dense(self, grad, var):
        var = var.handle
        beta1_power = math_ops.cast(self._beta1_power, grad.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, grad.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m").handle
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v").handle
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

        # amsgrad
        vhat = self.get_slot(var, "vhat").handle
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)

        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse_shared(self, grad, var, indices, scatter_add):
        beta1_power = math_ops.cast(self._beta1_power, var.dtype.base_dtype)
        beta2_power = math_ops.cast(self._beta2_power, var.dtype.base_dtype)
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)

        lr = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta1_t)
        m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
        with ops.control_dependencies([m_t]):
            m_t = scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta2_t)
        v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
        with ops.control_dependencies([v_t]):
            v_t = scatter_add(v, indices, v_scaled_g_values)

        # amsgrad
        vhat = self.get_slot(var, "vhat")
        vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
        v_sqrt = math_ops.sqrt(vhat_t)
        var_update = state_ops.assign_sub(var, lr * m_t / (v_sqrt + epsilon_t), use_locking=self._use_locking)
        return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

    def _apply_sparse(self, grad, var):
        return self._apply_sparse_shared(
            grad.values, var, grad.indices,
            lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
                x, i, v, use_locking=self._use_locking))

    def _resource_scatter_add(self, x, i, v):
        with ops.control_dependencies(
                [resource_variable_ops.resource_scatter_add(x.handle, i, v)]):
            return x.value()

    def _resource_apply_sparse(self, grad, var, indices):
        return self._apply_sparse_shared(
            grad, var, indices, self._resource_scatter_add)

    def _finish(self, update_ops, name_scope):
        # Update the power accumulators.
        with ops.control_dependencies(update_ops):
            with ops.colocate_with(self._beta1_power):
                update_beta1 = self._beta1_power.assign(
                    self._beta1_power * self._beta1_t,
                    use_locking=self._use_locking)
                update_beta2 = self._beta2_power.assign(
                    self._beta2_power * self._beta2_t,
                    use_locking=self._use_locking)
        return control_flow_ops.group(*update_ops + [update_beta1, update_beta2],
                                      name=name_scope)
