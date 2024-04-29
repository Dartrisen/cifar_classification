import os
import pickle

import numpy as np
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


# noinspection PyTypeChecker
def save_images(data: str, directory: str):
    num_images = len(data[b'data'])

    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in range(num_images):
        image_data = np.reshape(data[b'data'][i], (3, 32, 32))
        image_data = np.transpose(image_data, (1, 2, 0))

        # Extract label information
        image_label = data[b'labels'][i]

        # Create subdirectory for each label if it doesn't exist
        label_dir = os.path.join(directory, f'category_{image_label}')
        if not os.path.exists(label_dir):
            os.makedirs(label_dir)

        # Save image with appropriate filename
        image_filename = f'image_{i}__label_{image_label}.jpg'
        image_path = os.path.join(label_dir, image_filename)
        image = Image.fromarray(image_data)
        image.save(image_path)
        print(f'Saved image {i} to {image_path}')


def main():
    path = "path/to/dir"
    data_batch = unpickle(path)
    output_dir = 'train'
    save_images(data_batch, output_dir)


if __name__ == '__main__':
    main()
