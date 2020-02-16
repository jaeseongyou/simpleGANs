import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()
# takes [logit] NOT sigmoid outputs

def discriminator_loss(real_output, fake_output):
    real_loss = mse(tf.ones_like(real_output), real_output)
    fake_loss = mse(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return mse(tf.ones_like(fake_output), fake_output)
