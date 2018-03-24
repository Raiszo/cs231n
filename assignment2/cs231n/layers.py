from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    D = np.prod(x.shape[1:]) # d_1 * ... * d_k
    N = x.shape[0] # number of samples "N"
    xr = x.reshape(( N, D )) # reshaped as (N, D)

    out = xr.dot(w) + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b) # return x, not xr
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    D = np.prod(x.shape[1:])
    N = x.shape[0]
    xr = x.reshape(( N, D ))

    dw = xr.T.dot(dout)

    dxr = dout.dot(w.T)
    dx = dxr.reshape(x.shape) # do not forget the dimensions (N, d1, ..., d_k)

    db = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = np.maximum(0,x) # quite simple :D
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    # print(x.mean())
    # print(dout.mean())
    dx = (x >= 0) * dout # only valus x > 0 recieve a gradient :3
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        # do some statistics
        # mini_mean = x.mean(axis=0)
        # mini_var = x.var(axis=0)

        # store this, dunno why yet
        # running_mean = momentum * running_mean + (1 - momentum) * mini_mean
        # running_var = momentum * running_var + (1 - momentum) * mini_var

        # normalize and then shift to a desire mean and variance,
        # think of this to move the variance and mean to a desired value for
        # some reason I dunn
        # x_norm = (x - mini_mean) / np.sqrt(mini_var + eps)
        # out = gamma * x_norm + beta

        # start with x
        mean = x.mean(axis = 0)
        x_zero_mean = x - mean

        var = x.var(axis = 0)
        var_sqrt = np.sqrt(var + eps)
        var_inv = 1 / var_sqrt
        x_norm = x_zero_mean  * var_inv

        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var
        
        out = gamma * x_norm + beta
        cache = {
            'x': x,
            'mean': mean,
            'x_zero_mean': x_zero_mean,
            'var': var,
            'var_sqrt': var_sqrt,
            'var_inv': var_inv,
            'x_norm': x_norm,
            'beta': beta,
            'gamma': gamma,
            'eps': eps
        }
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_norm = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    
    beta = cache['beta']
    gamma = cache['gamma']

    x = cache['x']
    mean = cache['mean']
    x_zero_mean = cache['x_zero_mean']
    var = cache['var']
    var_sqrt = cache['var_sqrt']
    var_inv = cache['var_inv']
    x_norm = cache['x_norm']
    eps = cache['eps']

    N = x.shape[0]

    
    dx_norm = gamma * dout
    
    dx_zero_mean1 = var_inv * dx_norm
    
    # Dunno why, similar to b
    # It's because there is a substraction accross features (columns)
    # one does not see it coz of broadcasting
    dvar_inv = np.sum(x_zero_mean * dx_norm, axis=0)
    dvar_sqrt = -1 / (var_sqrt ** 2 + eps) * dvar_inv
    dvar = 0.5 / np.sqrt(var + eps) * dvar_sqrt
    
    dx_zero_mean2 = (2 * x_zero_mean) * np.ones_like(x) * (1/N) * dvar

    
    dx_zero_mean = dx_zero_mean1 + dx_zero_mean2


    # This is a summation node
    dx1 = dx_zero_mean
    dmean = - np.sum(dx_zero_mean, axis = 0)
    dx2 = 1/N * np.ones_like(x) * dmean

    dx = dx1 + dx2
    dgamma = np.sum(x_norm * dout, axis=0)
    dbeta = np.sum(dout, axis=0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # Use the * trick to unpack the shape tuple to be able to use it
        # as an argument
        mask = np.random.randn(*x.shape) >= p
        out = x * mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = dout * mask
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    padding	= conv_param['pad']
    stride	= conv_param['stride']
    N,C,H,W	= x.shape
    f,_,hh,ww	= w.shape


    out_h = 1+(H + 2*padding - hh) // stride
    out_w = 1+(W + 2*padding - ww) // stride
    out = np.zeros((N,f, out_h, out_w))
    half_h = hh//2 - (1 if hh%2 == 0 else 0)
    half_w = ww//2 - (1 if ww%2 == 0 else 0)

    # Go through all images
    for m in range(N):
        img = x[m]
        # add padding
        # first tuple is to not add padding accross the channels dim
        padded = np.pad(img, ((0,0), (padding,padding), (padding,padding)), 'constant')

        for i in range(out_h):
            for j in range(out_w):
                # start at zero, and move with stride
                # assuming stride = pad, bad D:x 
                current_h = (i*stride) if i>0 else 0
                current_w = (j*stride) if j>0 else 0
                # Once in a position, get a small window from the image
                mask = padded[:,current_h:current_h+hh, current_w:current_w+ww]

                # neat bradcasting :3
                prod = mask * w
                # Use this to sum elements across all dims, except for 0 (f)
                conv = prod.reshape(prod.shape[0], np.prod(prod.shape[1:])).sum(axis=1)
                out[m,:,i,j] = conv + b
        
        
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    (x, w, b, conv_param) = cache
    padding	= conv_param['pad']
    stride	= conv_param['stride']
    
    N,C,H,W	= x.shape
    f,_,hh,ww	= w.shape

    _,f,out_h,out_w = dout.shape
    half_h = hh//2 - (1 if hh%2 == 0 else 0)
    half_w = ww//2 - (1 if ww%2 == 0 else 0)

    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    dx = np.zeros_like(x)

    w_summed = w.reshape(w.shape[0], np.prod(w.shape[1:])).sum(axis=1)
    for m in range(N):
        img = x[m]
        # add padding
        # first tuple is to not add padding accross the channels dim
        padded = np.pad(img, ((0,0), (padding,padding), (padding,padding)), 'constant')
        dpadded = np.zeros_like(padded)
        for i in range(out_h):
            for j in range(out_w):
                dout_ij = dout[m,:,i,j] # shape f: number of filters
                db += dout_ij # bias term updates directly

                # Need to get the region of spational interest
                current_h = (i*stride) if i>0 else 0
                current_w = (j*stride) if j>0 else 0
                # Once in a position, get a small window from the image
                mask = padded[:,current_h:current_h+hh, current_w:current_w+ww]

                # expand each result from each filter convolution by
                # multiplying for a ONES matrix, the reshape is just a trick
                dout_conv_row = dout_ij.reshape(f,1).dot(np.ones((1,C*hh*ww)))
                dout_conv = dout_conv_row.reshape(f,C,hh,ww)

                # Once dout is projected, update dw
                # Neat broadcasting :3
                dw += mask*dout_conv

                dx_space = (w*dout_conv).sum(axis=0)
                dpadded[:,current_h:current_h+hh, current_w:current_w+ww] += dx_space
        # remember to cut off the padding
        dx[m] = dpadded[:,1:-1,1:-1]
                

    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    ph		= pool_param['pool_height']
    pw		= pool_param['pool_width']
    stride	= pool_param['stride']
    N,C,H,W	= x.shape


    out_shape = N, C, H//stride, W//stride
    out = np.zeros(out_shape)
    ind = np.zeros(out_shape)
    half_h = ph//2 - (1 if ph%2 == 0 else 0)
    half_w = pw//2 - (1 if pw%2 == 0 else 0)

    for m in range(N):
        img = x[m]

        for i in range(out.shape[2]):
            for j in range(out.shape[3]):
                current_h = (i*stride) if i>0 else 0
                current_w = (j*stride) if j>0 else 0
                mask = img[:, current_h:current_h+stride, current_w:current_w+stride]

                # reshape trick to get the max value easily
                maxis = mask.reshape(mask.shape[0], np.prod(mask.shape[1:]))
                out[m,:,i,j] = np.amax(maxis, axis=1)
                ind[m,:,i,j] = np.argmax(maxis, axis=1)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param, ind)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param, max_indexes = cache

    ph		= pool_param['pool_height']
    pw		= pool_param['pool_width']
    stride	= pool_param['stride']
    N,C,H,W	= x.shape


    dx = np.zeros_like(x)
    half_h = ph//2 - (1 if ph%2 == 0 else 0)
    half_w = pw//2 - (1 if pw%2 == 0 else 0)

    for m in range(N):
        for i in range(dout.shape[2]):
            for j in range(dout.shape[3]):
                # for each pooling result, get the position of the max value
                mask_row = np.zeros((C,ph*pw))
                mask_row[np.arange(C),max_indexes[m,:,i,j].astype(int)] = dout[m,np.arange(C),i,j]
                mask = mask_row.reshape(C,ph,pw)
                
                # Then add the gradient to the result (dx)
                dx[m,:,i*stride:(i+1)*stride,j*stride:(j+1)*stride] += mask

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N,C,H,W = x.shape

    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    # For some reason, need to add this shit
    gamma_b = gamma[None,:,None,None]
    beta_b = beta[None,:,None,None]
    
    running_mean = bn_param.get('running_mean', np.zeros(C, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(C, dtype=x.dtype))


    if mode == 'train':
        # One can do mean accross multiple dimensions -.-
        # I was reshaping and doing weird stuff all the time
        # 1 is the C dim
        mean = x.mean(axis=(0,2,3)) # shape (C,)
        x_zero_mean = x - mean[None,:,None,None] # Need to broadcast it manually
        
        var = x.var(axis=(0,2,3)) # shape (C,)
        var_sqrt = np.sqrt(var[None,:,None,None] + eps) # Need to broadcast it manually
        var_inv = 1 / var_sqrt
        x_norm = x_zero_mean * var_inv

        # running_mean and var keep the (C,) dimension
        running_mean = momentum * running_mean + (1 - momentum) * mean
        running_var = momentum * running_var + (1 - momentum) * var

        # Above is the manual broadcasting
        out = gamma_b * x_norm + beta_b
        cache = {
            'x': x,
            'mean': mean,
            'x_zero_mean': x_zero_mean,
            'var': var,
            'var_sqrt': var_sqrt,
            'var_inv': var_inv,
            'x_norm': x_norm,
            'beta': beta,
            'gamma': gamma,
            'eps': eps
        }
        
    elif mode == 'test':
        running_mean_add = running_mean[None,:,None,None]
        running_var_add = running_var[None,:,None,None]
        x_norm = (x - running_mean_add) / np.sqrt(running_var_add + eps)
        out = gamma_b * x_norm + beta_b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var
    
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################

    beta = cache['beta'][None,:,None,None]
    gamma = cache['gamma'][None,:,None,None]

    x = cache['x']
    mean = cache['mean'][None,:,None,None]
    x_zero_mean = cache['x_zero_mean']
    var = cache['var'][None,:,None,None]
    var_sqrt = cache['var_sqrt']
    var_inv = cache['var_inv']
    x_norm = cache['x_norm']
    eps = cache['eps']

    N,C,H,W = x.shape

    # print(dout.shape)
    # print(gamma.shape)
    dx_norm = gamma * dout
    # print(dx_norm.shape)
    
    dx_zero_mean1 = var_inv * dx_norm

    dvar_inv = np.sum(x_zero_mean * dx_norm, axis=(0,2,3), keepdims=True)
    # print(dvar_inv.shape)
    # print(var_sqrt.shape)
    dvar_sqrt = -1 / (var_sqrt ** 2 + eps) * dvar_inv
    # print(dvar_sqrt.shape)
    dvar = 0.5 / np.sqrt(var + eps) * dvar_sqrt
    # print(var.shape)
    
    dx_zero_mean2 = (2 * x_zero_mean) * np.ones_like(x) * (1/(N*H*W)) * dvar

    
    dx_zero_mean = dx_zero_mean1 + dx_zero_mean2


    # This is a summation node
    dx1 = dx_zero_mean
    dmean = - np.sum(dx_zero_mean, axis = (0,2,3), keepdims=True)
    dx2 = 1/(N*H*W) * np.ones_like(x) * dmean

    dx = dx1 + dx2
    dgamma = np.sum(x_norm * dout, axis=(0,2,3))
    dbeta = np.sum(dout, axis=(0,2,3))
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    # x is x.dot(w) already
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    # x is x.dot(w) already
    shifted_logits = x - np.max(x, axis=1, keepdims=True) # can do the shitfint first O:
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N # can do the log before
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
