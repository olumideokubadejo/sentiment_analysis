from data import load_mosi, extract_audio_data, extract_vision_data, extract_text_data, \
    extract_labels_data
from models import contextual_attention_model
from train import TrainMultiModal

if __name__ == '__main__':
    data = load_mosi()
    train_a, val_a, test_a = extract_audio_data(data)
    train_v, val_v, test_v = extract_vision_data(data)
    train_t, val_t, test_t = extract_text_data(data)
    train_l, val_l, test_l = extract_labels_data(data)

    model = contextual_attention_model(train_t, train_a, train_v)
    trainer = TrainMultiModal(model)
    trainer.train(train_t, train_a, train_v, train_l, val_t, val_a, val_v, val_l,)
    trainer.test(test_t, test_a, test_v, test_l)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
