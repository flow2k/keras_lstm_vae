import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector
from tensorflow.keras.layers import Flatten, Dense, Dropout, Lambda
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
# from tensorflow.keras import objectives
from tensorflow.keras.losses import mse, binary_crossentropy


def create_lstm_vae(input_dim,
                    timesteps,
                    batch_size,
                    intermediate_dim,
                    latent_dim,
                    epsilon_std=1.):
    """
    Creates an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator.

    # Arguments
        input_dim: int.
        timesteps: int, input timestep dimension.
        batch_size: int.
        intermediate_dim: int, output shape of LSTM.
        latent_dim: int, latent z-layer shape.
        epsilon_std: float, z-layer sigma.


    # References
        - [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        - [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,))
    # x2 = tf.Print(x, [x], message="EncInput", summarize=1024)
    # h = LSTM(intermediate_dim)(x2)
    h = LSTM(intermediate_dim)(x)

    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

    # decoded LSTM layer
    # decoder_h = LSTM(intermediate_dim, return_sequences=True)
    # decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    # h_decoded = decoder_h(h_decoded)
    # h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    x_decoded_mean = Dense(input_dim, activation=None)(x_decoded_mean)

    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    # _h_decoded = decoder_h(_h_decoded)
    # _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)

    # def vae_loss(x, x_decoded_mean):
    #     xent_loss = mse(x, x_decoded_mean)
    #     # xent_loss = K.print_tensor(xent_loss, message="Losss")
    #     # xent_loss = tf.Print(xent_loss, [xent_loss], message = "TF Losss", summarize = 1024)
    #     kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
    #     loss = xent_loss + kl_loss
    #     return loss

    xent_loss = mse(x, x_decoded_mean)
    # xent_loss = K.print_tensor(xent_loss, message="Losss")
    # xent_loss = tf.Print(xent_loss, [xent_loss], message = "TF Losss", summarize = 1024)
    kl_loss = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
    vae_loss2 = xent_loss + kl_loss
    vae_loss2 = K.mean(vae_loss2)
    # vae_loss2 = K.print_tensor(vae_loss2, message="Losss")

    # # vae.compile(optimizer='rmsprop', loss=vae_loss)
    #     vae.add_loss(vae_loss2)
    #     vae.compile(optimizer='rmsprop')

    # vae.compile(optimizer='adam', loss=vae_loss)
    vae.add_loss(vae_loss2)
    vae.compile(optimizer='adam')

    print("generated model with Dense")
    return vae, encoder, generator

