from keras.layers import Layer, Masking, Bidirectional, GRU, Dropout, Dense, \
    TimeDistributed


class RecurrentEmbedding(Layer):
    def __init__(self, mask_value=0, drop_rnn=0.7, gru_units=300, drop_dense=0.7,
                 dense_units=100, dropout=0.2, recurrent_dropout=0.3):
        super(RecurrentEmbedding, self).__init__()
        self.mask_value = mask_value
        self.drop_rnn = drop_rnn
        self.gru_units = gru_units
        self.drop_dense = drop_dense
        self.dense_units = dense_units
        self.dropout = dropout
        self.recurrent_dropout = recurrent_dropout
        self.mask_layer = Masking(mask_value=mask_value)
        self.lstm = Bidirectional(
            GRU(gru_units, return_sequences=True, dropout=dropout,
                recurrent_dropout=recurrent_dropout),
            merge_mode='concat')
        self.rnn_dropout = Dropout(drop_rnn)
        self.dense_dropout = Dropout(drop_dense)
        self.time_distributed = TimeDistributed(Dense(dense_units, activation='tanh'))

    # def build(self, input_shape):
    #     self.input = Input(shape=(input_shape[1], input_shape[2]))

    def call(self, data):
        output = self.mask_layer(data)
        output = self.lstm(output)
        output = self.rnn_dropout(output)
        output = self.dense_dropout(self.time_distributed(output))
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "mask_value": self.mask_value,
            "drop_rnn": self.drop_rnn,
            "gru_units": self.gru_units,
            "drop_dense": self.drop_dense,
            "dense_units": self.dense_units,
            "dropout": self.dropout,
            "recurrent_dropout": self.recurrent_dropout
        })
        return config
