from keras.layers import Layer, Dense, TimeDistributed, Conv1D, Flatten


class PostFusion(Layer):
    def __init__(self):
        super(PostFusion, self).__init__()
        self.time_distributed = TimeDistributed(Dense(2, activation='softmax'))
        self.conv1 = Conv1D(50, 11)
        self.conv2 = Conv1D(25, 11)
        self.flatten = Flatten()
        self.dense1 = Dense(25)
        self.dense2 = Dense(1)

    def call(self, fused_encoding):
        output = self.time_distributed(fused_encoding)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.flatten(output)
        output = self.dense1(output)
        output = self.dense2(output)
        return output
