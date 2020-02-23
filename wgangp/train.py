import tensorflow as tf
from model import *
from data import *
from loss import *
import os

def train_step_gen():
	with tf.GradientTape() as tape:
		z = tf.random.normal([BATCH_SIZE, NOISE_DIM])
		#z = tf.random.uniform([BATCH_SIZE, NOISE_DIM], -1.0, 1.0)
		fake_sample = generator(z)
		fake_score = discriminator(fake_sample)
		loss = generator_loss(fake_score)
	gradients = tape.gradient(loss, generator.trainable_variables)
	generator_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
	return loss

def train_step_dis(real_sample):
	with tf.GradientTape() as tape:
		z = tf.random.normal([BATCH_SIZE, NOISE_DIM])
		#z = tf.random.uniform([BATCH_SIZE, NOISE_DIM], -1.0, 1.0)
		fake_sample = generator(z)
		real_score = discriminator(real_sample)
		fake_score = discriminator(fake_sample)
		loss = discriminator_loss(real_score, fake_score)

		alpha = tf.random.uniform([BATCH_SIZE, 1, 1, 1], 0.0, 1.0)
		inter_sample = fake_sample * alpha + real_sample * (1 - alpha)
		with tf.GradientTape() as tape_gp:
			tape_gp.watch(inter_sample)
			inter_score = discriminator(inter_sample)
		gp_gradients = tape_gp.gradient(inter_score, inter_sample)
		gp_gradients_norm = tf.sqrt(tf.reduce_sum(tf.square(gp_gradients), axis = [1, 2, 3]))
		gp = tf.reduce_mean((gp_gradients_norm - 1.0) ** 2)
		loss += gp * 10.
	gradients = tape.gradient(loss, discriminator.trainable_variables)
	discriminator_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
	return loss

if __name__ == '__main__':

	EPOCH = 100
	BUFFER_SIZE = 5000
	BATCH_SIZE = 16
	NOISE_DIM = 100
	CKPT_DIR = 'ckpts'
	LOG_DIR = 'logs'

	train_images = make_celeba_dataset()
	dataset = train_images.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

	generator = Generator(NOISE_DIM)
	discriminator = Discriminator()

	generator_optimizer = tf.keras.optimizers.Adam(1e-5)
	discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)

	ckpt_prefix = os.path.join(CKPT_DIR, "ckpt")
	checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
	discriminator_optimizer=discriminator_optimizer,
	generator=generator,
	discriminator=discriminator)

	train_summary_writer = tf.summary.create_file_writer(LOG_DIR)

	#z = tf.random.uniform([BATCH_SIZE, NOISE_DIM], -1.0, 1.0)
	z = tf.random.normal([BATCH_SIZE, NOISE_DIM])
	cnt = 0
	for epoch in range(EPOCH):
		fake_sample = generator(z)
		with train_summary_writer.as_default():
			tf.summary.image("generated_image", fake_sample, step=cnt, max_outputs=9)

		for batch in dataset:
			cnt += 1
			for _ in range(5):
				dis_loss = train_step_dis(batch)

			gen_loss = train_step_gen()
			template = 'Epoch {}, Gen Loss: {}, Dis Loss: {}'
			print (template.format(epoch + 1, gen_loss, dis_loss))
			with train_summary_writer.as_default():
				tf.summary.scalar('generator_loss', gen_loss, step=cnt)
				tf.summary.scalar('discriminator_loss', dis_loss, step=cnt)
