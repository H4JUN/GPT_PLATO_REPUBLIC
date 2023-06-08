import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Layer, Dense, Dropout, Add, LayerNormalization
from tensorflow.experimental.numpy import tril
from keras.losses import SparseCategoricalCrossentropy
import numpy as np
import pickle


with open("str_to_int", "rb") as f:
    str_to_int = pickle.load(f)

with open("int_to_str", "rb") as f:
    int_to_str = pickle.load(f)
    
# hyperparameters
batch_size = 64
block_size = 256 # How many items in the byte pair encoding to consider
vocab_size = len(str_to_int)
dropout_rate = 0.2
n_embd = 256 # Every head is now 256 / 4 = 64
n_head = 4
n_layer = 4

def positional_encoding(length, depth):
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]     # (seq, 1)
    depths = np.arange(depth)[np.newaxis, :]/depth   # (1, depth)

    angle_rates = 1 / (10000**depths)         # (1, depth)
    angle_rads = positions * angle_rates      # (pos, depth)

    pos_encoding = np.concatenate(
      [np.sin(angle_rads), np.cos(angle_rads)],
      axis=-1) 

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, n_embd) 
        self.pos_encoding = positional_encoding(length=block_size, depth=n_embd)

    def call(self, x):
        length = tf.shape(x)[1]
        x = self.embedding(x)
        # This factor sets the relative scale of the embedding and positonal_encoding.
        x *= tf.math.sqrt(tf.cast(n_embd, tf.float32))
        x = x + self.pos_encoding[tf.newaxis, :length, :]
        return x
    
    
class FeedForward(Layer):
    def __init__(self, n_embd):
        super().__init__()
        self.seq = Sequential([
          Dense(4*n_embd, activation='relu'),
          Dense(n_embd),
          Dropout(dropout_rate)
        ])
        self.add = Add() #################### Different from pytorch
        self.layer_norm = LayerNormalization()

    def call(self, x):
        x = self.add([x, self.seq(x)])
        x = self.layer_norm(x) 
        return x
    
class Head(Layer):
    def __init__(self, head_size):
        super().__init__()
        self.key = Dense(head_size, input_shape=(n_embd,), use_bias=False)
        self.query = Dense(head_size, input_shape=(n_embd,), use_bias=False)
        self.value = Dense(head_size, input_shape=(n_embd,), use_bias=False)
        self.dropout = Dropout(dropout_rate)
        self.mask = tril(tf.ones(shape=(block_size, block_size)))
        
    def call(self, x):
        B, T, C = x.shape
        k = self.key(x) # What do I contain
        q = self.query(x) # What am I looking for?
        # scaled attention: divide by square root of head_size. Otherwise, the variance of weights will be too high
        # This is important since when applying softmax, the weights hsould be fairly well-distributed.
        # If the weights have high variance, then softmax will ocnverge to be one-hot vectors
        weights = tf.matmul(q, k, transpose_b=True) * C**-0.5 # For each token, comapre with other tokens (including itself, hence self-attention) and see the one that resonates best
        weights = tf.where(self.mask[:T, :T]==0, -np.inf, weights) # If this is confusing, check 48:30 from GPT video
        weights = tf.nn.softmax(weights)
        weights = self.dropout(weights)
        
        v = self.value(x)
        output = tf.matmul(weights, v) # size is (B, T, T) X (B, T, C) = (B, T, C)
        return output
    
class MultiHeadAttention(Layer):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = [Head(head_size) for _ in range(num_heads)]
        self.projection = Dense(n_embd, input_shape=(n_embd,))
        self.dropout = Dropout(dropout_rate)
        
    def call(self, x):
        output = tf.concat([h(x) for h in self.heads], -1)
        output = self.dropout(self.projection(output))
        return output
    
class Block(Layer):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size)
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm_1 = LayerNormalization()
        self.layer_norm_2 = LayerNormalization()
        self.add = Add()
        
    def call(self, x):
        x = self.add([x, self.self_attention(self.layer_norm_1(x))])
        x = self.add([x, self.feed_forward(self.layer_norm_2(x))])
        return x
    
class Multinomial_Model(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.positional_embedding = PositionalEmbedding()
        self.blocks = Sequential([Block(n_embd, n_head) for _ in range(n_layer)])
        self.layer_norm = LayerNormalization()
        self.dense = Dense(vocab_size)
        self.loss_fn = SparseCategoricalCrossentropy(from_logits=True)
        
    def call(self, x, y=None):
        x = self.positional_embedding(x)
        x = self.blocks(x)
        x = self.layer_norm(x)
        logits = self.dense(x)
        
        if y is None:
            loss = None
        else:
            batch, timestep, channel = logits.shape
            logits = tf.reshape(logits, [batch*timestep, channel])
            y = tf.reshape(y, [batch*timestep])
            loss = self.loss_fn(y, logits)
        return logits, loss
    
    def generate(self, x, max_new_tokens):
        if x.shape[0] < block_size: # If input shape is small
            diff = block_size - x.shape[0]
            x = tf.pad(x, [[0, 0], [diff, 0]], "CONSTANT", constant_values=1) # Pad left 
        for _ in range(max_new_tokens):
            x_cropped = x[:, -block_size:] # in case x has more timesteps(tokens) than the block_size
            logits, loss = self(x_cropped)
            logits = logits[:, -1, :] # Becomes (B, C) Get the last prediction
            probabilities = tf.nn.softmax(logits)
            x_next = tf.random.categorical(tf.math.log(probabilities), 1, dtype=tf.int32)
            x = tf.concat([x, x_next], axis=1)
        return x