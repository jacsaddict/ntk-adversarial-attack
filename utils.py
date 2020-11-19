import numpy as onp
import jax.numpy as np
from jax.experimental import optimizers
from jax.api import grad, jit, vmap
from jax import lax, random
from jax.config import config
config.update('jax_enable_x64', True)

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import neural_tangents as nt

def format_plot(title='', x='', y='', grid=True):  
    ax = plt.gca()

    plt.grid(grid)
    if title:
        plt.title(title, fontsize=26)
    plt.xlabel(x, fontsize=22)
    plt.ylabel(y, fontsize=22)

def finalize_plot(shape=(1, 1)):
    plt.gcf().set_size_inches(
        shape[0] * 1.5 * plt.gcf().get_size_inches()[1], 
        shape[1] * 1.5 * plt.gcf().get_size_inches()[1])
    plt.tight_layout()

def plot_fn(train, test, *fs):
    train_xs, train_ys = train
    plt.plot(train_xs, train_ys, 'ro', markersize=10, label='train')

    if test != None:
        test_xs, test_ys = test
        plt.plot(test_xs, test_ys, 'k--', linewidth=3, label='$f(x)$')

        for f in fs:
            plt.plot(test_xs, f(test_xs), '-', linewidth=3)

    plt.xlim([-np.pi, np.pi])
    plt.ylim([-1.5, 1.5])
    format_plot('$x$', '$f$')

# Kernel Construction
_Kernel = nt.utils.kernel.Kernel

def Kernel(K):
    """Create an input Kernel object out of an np.ndarray."""
    return _Kernel(cov1=np.diag(K), nngp=K, cov2=None, 
                   ntk=None, is_gaussian=True, is_reversed=False,
                   diagonal_batch=True, diagonal_spatial=False,
                   batch_axis=0, channel_axis=1, mask1=None, mask2=None,
                   shape1=(2, 1024), shape2=(2,1024),
                   x1_is_x2=True, is_input=True) 

def NTKernel(var1, nngp, var2, ntk):
    """Create an input Kernel object out of an np.ndarray."""
    return _Kernel(cov1=var1, nngp=nngp, cov2=var2, 
                   ntk=ntk, is_gaussian=True, is_reversed=False,
                   diagonal_batch=True, diagonal_spatial=False,
                   batch_axis=0, channel_axis=1, mask1=None, mask2=None,
                   shape1=(2, 1024), shape2=(2,1024),
                   x1_is_x2=True, is_input=True) 

def wrap(kernel_fn):
    def wrapped_fn(kernel):
        out = kernel_fn(NTKernel(*kernel))
        return kernel._replace(cov1=out.cov1, nngp=out.nngp, cov2=out.cov2, ntk=out.ntk)
    return wrapped_fn

def fixed_point(f, initial_value, threshold):
    """Find fixed-points of a function f:R->R using Newton's method."""
    g = lambda x: f(x) - x
    dg = grad(g)

    def cond_fn(x):
        x, last_x = x
        return np.abs(x - last_x) > threshold

    def body_fn(x):
        x, _ = x
        return x - g(x) / dg(x), x
    return lax.while_loop(cond_fn, body_fn, (initial_value, 0.0))[0]

# TODO: This is necessary because of a bug in NT's CPU detection inside a jit
nt.predict._arr_is_on_cpu = lambda x: False

def scale(a, b):
    return a * b[-1] / a[-1]

def normalize(x_train, y_train, x_test, y_test):
    x_train = x_train / np.sqrt(np.reshape(np.einsum('ij,ij->i', x_train, x_train), (64, 1))) * np.sqrt(x_train.shape[-1])
    y_train = y_train - np.mean(y_train, axis=0, keepdims=True)
    x_test = x_test / np.sqrt(np.reshape(np.einsum('ij,ij->i', x_test, x_test), (32, 1))) * np.sqrt(x_test.shape[-1])
    return x_train, y_train, x_test, y_test

# Data Loading
def _partial_flatten_and_normalize(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    x = onp.reshape(x, (x.shape[0], -1))
    return (x - onp.mean(x)) / onp.std(x)

def _partial_flatten(x):
    """Flatten all but the first dimension of an `np.ndarray`."""
    return onp.reshape(x, (x.shape[0], -1))/255

def _one_hot(x, k, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""
    return onp.array(x[:, None] == onp.arange(k), dtype)

def get_dataset(name, n_train=None, n_test=None, permute_train=False,
                do_flatten_and_normalize=True):
    """Download, parse and process a dataset to unit scale and one-hot labels."""
    ds_builder = tfds.builder(name)

    ds_train, ds_test = tfds.as_numpy(
        tfds.load(
            name + ':3.*.*',
            split=['train' + ('[:%d]' % n_train if n_train is not None else ''),
                   'test' + ('[:%d]' % n_test if n_test is not None else '')],
            batch_size=-1,
            as_dataset_kwargs={'shuffle_files': False}))
    
    train_images, train_labels, test_images, test_labels = (ds_train['image'],
                                                            ds_train['label'],
                                                            ds_test['image'],
                                                            ds_test['label'])
    
    if do_flatten_and_normalize:
        train_images = _partial_flatten_and_normalize(train_images)
        test_images = _partial_flatten_and_normalize(test_images)
    else:
        train_images = _partial_flatten(train_images)
        test_images = _partial_flatten(test_images)

    num_classes = ds_builder.info.features['label'].num_classes
    train_labels = _one_hot(train_labels, num_classes)
    test_labels = _one_hot(test_labels, num_classes)

    if permute_train:
        perm = onp.random.RandomState(0).permutation(train_images.shape[0])
        train_images = train_images[perm]
        train_labels = train_labels[perm]

    return train_images, train_labels, test_images, test_labels

def show_images(image, num_row=2, num_col=5):
    # plot images
    image_size = int(onp.sqrt(image.shape[-1]))
    image = np.reshape(image, (image.shape[0], image_size, image_size))
    fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
    for i in range(num_row*num_col):
        ax = axes[i//num_col, i%num_col]
        ax.imshow(image[i], cmap='gray', vmin=0, vmax=1)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def shaffle(images, labels, seed=None):
    perm = onp.random.RandomState(seed).permutation(images.shape[0])
    images = images[perm]
    labels = labels[perm]
    return images, labels
