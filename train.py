from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.losses import mean_absolute_error
from keras.models import Model


class TrainMultiModal:
    def __init__(self, model: Model, epochs=200, optimizer="adam", learning_rate=0.0001,
                 batch_size=32, save_path=None):
        self.model = model
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        save_path = 'weights/Mosi_Trimodal_' + '_Run_' + '.hdf5' if not save_path else save_path
        self.save_path = save_path

    def train(self, train_text, train_audio, train_video, train_label, dev_text,
              dev_audio, dev_video, dev_label):
        self.model.compile(optimizer='adam', loss='mae',
                           sample_weight_mode='temporal', metrics=['mse'])

        early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=0)
        check = ModelCheckpoint(self.save_path, monitor='val_loss', save_best_only=True,
                                mode='max', verbose=0)
        # train model #
        history = self.model.fit([train_text, train_audio, train_video], train_label,
                                 epochs=self.epochs,
                                 batch_size=self.batch_size,
                                 # sample_weight=train_mask,
                                 shuffle=True,
                                 callbacks=[early_stop, check],
                                 validation_data=(
                                     [dev_text, dev_audio, dev_video], dev_label),
                                 verbose=1)

    def test(self, test_text, test_audio, test_video, test_label):
        test_predictions = self.model.predict([test_text, test_audio, test_video])
        return mean_absolute_error(test_predictions, test_label)
