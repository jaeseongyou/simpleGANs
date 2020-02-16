import matplotlib.pyplot as plt
import os


def generate_images(model, test_input):
    predictions = model(test_input, training=False)
    return predictions

def generate_and_save_images(path, model, epoch, test_input):
    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4,4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, :] *.5 + .5)
        plt.axis('off')
    plt.savefig(os.path.join(path,'image_at_epoch_{:04d}.png').format(epoch))
