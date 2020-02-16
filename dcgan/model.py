import tensorflow as tf


class Generator(tf.keras.Model):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(4*4*1024, use_bias=False, input_shape=(z_dim,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.Reshape((4, 4, 1024)),

            tf.keras.layers.Conv2DTranspose(512, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            tf.keras.layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            tf.keras.layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            tf.keras.layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ])

    def call(self, z):
        x = self.net(z)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = tf.keras.Sequential([
            tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same', input_shape=(64, 64, 3)),
            tf.keras.layers.LeakyReLU(alpha=0.2),

            tf.keras.layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Conv2D(1024, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(alpha=0.2),
            tf.keras.layers.BatchNormalization(),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    def call(self, image):
        x = self.net(image)
        return x
