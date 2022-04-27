from keras.layers import Input, concatenate
from keras.models import Model

from layers.attention import BiModalAttention
from layers.embeddings import RecurrentEmbedding
from layers.encoders import PostFusion


def contextual_attention_model(train_text, train_audio, train_video):
    in_text = Input(shape=(train_text.shape[1], train_text.shape[2]))
    in_audio = Input(shape=(train_audio.shape[1], train_audio.shape[2]))
    in_video = Input(shape=(train_video.shape[1], train_video.shape[2]))
    text_embedding = RecurrentEmbedding()(in_text)
    audio_embedding = RecurrentEmbedding()(in_audio)
    video_embedding = RecurrentEmbedding()(in_video)
    vt_att = BiModalAttention()(video_embedding, text_embedding)
    av_att = BiModalAttention()(audio_embedding, video_embedding)
    ta_att = BiModalAttention()(text_embedding, audio_embedding)

    merged = concatenate(
        [vt_att, av_att, ta_att, video_embedding, audio_embedding, text_embedding])

    output = PostFusion()(merged)
    model = Model([in_text, in_audio, in_video], output)

    return model
