import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from einops.layers.tensorflow import Rearrange
import math


# GELU activation

def gelu(x):
    cdf = 0.5 * (1.0 + tf.tanh(
        math.sqrt(2 / math.pi) * (x + 0.044715 * tf.pow(x, 3))))
    return x * cdf

def get_activation(name):
    if name.lower() == "gelu":
        return gelu
    return tf.keras.activations.get(name)


# Transformer components

class Residual(tf.keras.Model):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def call(self, x):
        return self.fn(x) + x

class PreNorm(tf.keras.Model):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.fn = fn
    def call(self, x):
        return self.fn(self.norm(x))

class FeedForward(tf.keras.Model):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=get_activation('gelu')),
            tf.keras.layers.Dense(dim)
        ])
    def call(self, x):
        return self.net(x)

class Attention(tf.keras.Model):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
        self.to_out = tf.keras.layers.Dense(dim)
        self.rearrange_qkv = Rearrange('b n (qkv h d) -> qkv b h n d', qkv=3, h=heads)
        self.rearrange_out = Rearrange('b h n d -> b n (h d)')
    def call(self, x):
        qkv = self.to_qkv(x)
        qkv = self.rearrange_qkv(qkv)
        q, k, v = qkv[0], qkv[1], qkv[2]
        dots = tf.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = tf.nn.softmax(dots, axis=-1)
        out = tf.einsum('bhij,bhjd->bhid', attn, v)
        out = self.rearrange_out(out)
        return self.to_out(out)

class Transformer(tf.keras.Model):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ])
        self.net = tf.keras.Sequential(layers)
    def call(self, x):
        return self.net(x)


# Vision Transformer

class ViT(tf.keras.Model):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'Image size must be divisible by patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size
        self.dim = dim
        self.pos_embedding = self.add_weight(
            shape=[1, num_patches + 1, dim],
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )
        self.cls_token = self.add_weight(
            shape=[1, 1, dim],
            initializer=tf.keras.initializers.RandomNormal(),
            dtype=tf.float32
        )
        self.patch_to_embedding = tf.keras.layers.Dense(dim)
        self.rearrange = Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size)
        self.transformer = Transformer(dim, depth, heads, mlp_dim)
        self.mlp_head = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_dim, activation=get_activation('gelu')),
            tf.keras.layers.Dense(num_classes)
        ])

    def call(self, img):
        batch_size = tf.shape(img)[0]
        x = self.rearrange(img)
        x = self.patch_to_embedding(x)
        cls_tokens = tf.broadcast_to(self.cls_token, (batch_size, 1, self.dim))
        x = tf.concat([cls_tokens, x], axis=1)
        x += self.pos_embedding
        x = self.transformer(x)
        x = x[:, 0]  # take cls token
        return self.mlp_head(x)


# Load CIFAR-10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
y_train = y_train.flatten()
y_test = y_test.flatten()

BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(BATCH_SIZE)


# Compile & Train

vit_config = dict(
    image_size=32,
    patch_size=4,
    num_classes=10,
    dim=64,
    depth=6,
    heads=8,
    mlp_dim=128,
)

model = ViT(**vit_config)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_dataset, validation_data=test_dataset, epochs=100)
