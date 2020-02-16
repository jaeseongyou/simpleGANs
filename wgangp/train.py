import tensorflow as tf
import numpy as np
from data import *
from model import *
from util import *
from loss import *
import os


@tf.function
def train_step(images, step):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

        if step % 5 == 0:
            gp = gradient_penalty(discriminator, images, generated_images)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss


def gradient_penalty(discriminator, images, generated_images):
    alpha = tf.random.uniform([], minval=0., maxval=1.)
    interpolated_images = alpha * images + (1 - alpha) * generated_images
    print('interpolated_images', interpolated_images.shape)
    with tf.GradientTape() as disc_tape:
        disc_tape.watch(interpolated_images)
        interpolated_output = discriminator(interpolated_images)
    print('interpolated_output', interpolated_output.shape)
    gradients = disc_tape.gradient(interpolated_output, [interpolated_images])[0]
    print('gradients', gradients.shape)
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1, 2, 3]))
    print('slopes', slopes.shape)
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp


if __name__ == '__main__':

    EPOCH = 100
    BUFFER_SIZE = 10000
    BATCH_SIZE = 16
    NOISE_DIM = 100

    CKPT_DIR = 'ckpts'
    IMG_DIR = 'images'
    LOG_DIR = 'logs'


    train_images = make_celeba_dataset()
    dataset = train_images.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

    generator = Generator(NOISE_DIM)
    discriminator = Discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    ckpt_prefix = os.path.join(CKPT_DIR, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                     discriminator_optimizer=discriminator_optimizer,
                                     generator=generator,
                                     discriminator=discriminator)

    train_summary_writer = tf.summary.create_file_writer(LOG_DIR)

    step = 0
    seed = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    for epoch in range(EPOCH):
        print('Epoch', epoch)
        for image_batch in dataset:
            step += 1
            gen_loss, disc_loss = train_step(image_batch, step)
            with train_summary_writer.as_default():
                tf.summary.scalar('gen_loss', gen_loss, step=step)
                tf.summary.scalar('disc_loss', disc_loss, step=step)

        #generate_and_save_images(IMG_DIR, generator, epoch + 1, seed)
        generated = generate_images(generator, seed)
        with train_summary_writer.as_default():
            tf.summary.image("generated_image", generated, step=step, max_outputs=16)

        if (epoch + 1) % 20 == 0:
          checkpoint.save(file_prefix = ckpt_prefix)
