import math
import tensorflow as tf
from einops.layers.tensorflow import Rearrange

def gelu(x):
    """Smoothened Gaussian Error Linear Unit activation function.
    args:
        x: input tensor
    returns:
        output tensor after applying gelu"""
    fn = 0.5 * x * (1.0 + tf.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return fn

class PatchExtract(tf.keras.layers.Layer):
    """Extracts patches from the input image"""
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.rearrange = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        
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

class FeedForward(tf.keras.layers.Layer):
    """Feed forward MLP for transformer block"""
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.ff = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.keras.activations.get(gelu)),
            tf.keras.layers.Dense(dim)
        ])
        
    def call(self, x):
        return self.ff(x)

class Attention(tf.keras.layers.Layer):
    """Multi-head self attention layer"""
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        
        self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim)
        
        self.rearrange_qkv = Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)
        self.rearrange_out = Rearrange('b h n d -> b n (h d)')
        
    def call(self, x):
        qkv = self.to_qkv(x)
        qkv = self.rearrange_qkv(qkv)
        
        q = qkv[0]
        k = qkv[1]
        v = qkv[2]
        
        mul = tf.einsum('bhid,bhjd->bhij',q, k) * self.scale
        attn = tf.nn.softmax(mul, axis=-1)  
        
        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        out = self.rearrange_out(out)
        out = self.to_out(out)
        
        return out

class Transformer(tf.keras.Model):
    """Transformer block consisting of attention and MLP layers"""

    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                Residual(PreNorm(dim, Attention(dim, heads = heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ])
        self.net = tf.keras.Sequential(layers)

    def call(self, x):
        return self.net(x)
    

class ViT(tf.keras.Model):
    """Vision Transformer model"""

    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim
        
        self.patch_extractor = PatchExtract(patch_size)
        self.patch_to_embedding = tf.keras.layers.Dense(dim)
        self.cls_token = self.add_weight(
            name='cls_token',
            shape=(1, 1, dim), 
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
            )
        
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(num_patches + 1, dim),
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
            )
        
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.to_cls_token = tf.keras.layers.Identity()
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation=tf.keras.activations.get(gelu)),
            tf.keras.layers.Dense(num_classes)
        ])
        
    def call(self, img):
        b = tf.shape(img)[0]
        x = self.patch_extractor(img)
        x = self.patch_to_embedding(x)

        cls_tokens = tf.broadcast_to(self.cls_token, [b, 1, self.dim])
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding
        
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)     
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             