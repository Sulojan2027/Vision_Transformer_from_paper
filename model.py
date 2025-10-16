import math
import tensorflow as tf
from einops.layers.tensorflow import Rearrange

def gelu(x):
    """Smoothened Gaussian Error Linear Unit activation function.
    args:
        x: input tensor
    returns:
        output tensor after applying gelu"""
    out = 0.5 * x * (1.0 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return out

class PatchExtract(tf.keras.layers.Layer):
    """Extracts patches from the input image"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.rearrange = Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        
    def call(self, img):
        return self.rearrange(img)

class Residual(tf.keras.layers.Layer):
    """Residual skip connection in Encoder"""
    def __init__(self, func):
        super().__init__()
        self.func = func
        
    def call(self, x):
        return self.func(x) + x
    
class PreNorm(tf.keras.layers.Layer):
    """Layer normalization before attention and MLP blocks"""
    def __init__(self, dim, func):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.func = func
        
    def call(self, x):
        return self.func(self.norm(x))

class MLP(tf.keras.layers.Layer):
    """Feed forward MLP for transformer block"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.keras.activations.get(gelu)),
            tf.keras.layers.Dense(dim)
        ])
        
    def call(self, x):
        return self.ff(x)


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           