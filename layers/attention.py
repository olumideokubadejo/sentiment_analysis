import tensorflow as tf
from keras.layers import Activation, dot, multiply, concatenate, Layer


class BiModalAttention(Layer):
    def __init__(self):
        super(BiModalAttention, self).__init__()

    def call(self, x: tf.Tensor, y: tf.Tensor):
        m1 = dot([x, y], axes=[2, 2])
        n1 = Activation('softmax')(m1)
        o1 = dot([n1, y], axes=[2, 1])
        a1 = multiply([o1, x])

        m2 = dot([y, x], axes=[2, 2])
        n2 = Activation('softmax')(m2)
        o2 = dot([n2, x], axes=[2, 1])
        a2 = multiply([o2, y])

        return concatenate([a1, a2])
