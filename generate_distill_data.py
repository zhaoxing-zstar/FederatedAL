from locale import normalize
import os
import argparse

import torch
import torchvision
import numpy as np

from pytorch_pretrained_biggan import BigGAN, truncated_noise_sample

import utils


def generate_distill_data(
    batch_size, num_images_per_classes, device, output_path
):
    generator_model = BigGAN.from_pretrained(
        "biggan-deep-128", cache_dir=os.path.join("../data/checkpoint", "cached_model")
    )
    generator_model = generator_model.to(device)

    # prepare a input
    truncation = 0.4
    if os.path.exists(output_path):
        utils.remove_folder(output_path)
    else:
        utils.build_dirs(f"{output_path}")
    for class_idx in range(1000):
        _id = 0
        num_batches = int(num_images_per_classes / batch_size)

        utils.build_dirs(f"{output_path}/{class_idx}")
        for _ in range(num_batches):
            class_vector = one_hot_from_int(class_idx, batch_size=batch_size)
            noise_vector = truncated_noise_sample(
                truncation=truncation, batch_size=batch_size
            )
            noise_vector = torch.from_numpy(noise_vector).to(device)
            class_vector = torch.from_numpy(class_vector).to(device)

            # generate images
            with torch.no_grad():
                generated_images = generator_model(
                    noise_vector, class_vector, truncation
                ).clamp(min=-1, max=1)

            for image in generated_images:
                torchvision.utils.save_image(
                    image,
                    fp=f"{output_path}/{class_idx}/{_id}",
                    format="JPEG",
                    scale_each=True,
                    normalize=True,
                )
                _id += 1
        print(f"Finished {class_idx + 1}/1000.")


def one_hot_from_int(int_or_list, batch_size=1):
    """Create a one-hot vector from a class index or a list of class indices.
    Params:
        int_or_list: int, or list of int, of the imagenet classes (between 0 and 999)
        batch_size: batch size.
            If int_or_list is an int create a batch of identical classes.
            If int_or_list is a list, we should have `len(int_or_list) == batch_size`
    Output:
        array of shape (batch_size, 1000)
    """
    if isinstance(int_or_list, int):
        int_or_list = [int_or_list]

    if len(int_or_list) == 1 and batch_size > 1:
        int_or_list = [int_or_list[0]] * batch_size

    assert batch_size == len(int_or_list)

    array = np.zeros((batch_size, 1000), dtype=np.float32)
    for i, j in enumerate(int_or_list):
        array[i, j] = 1.0
    return array


def main():
    # Genrating Settings
    parser = argparse.ArgumentParser(description="Generate Distillation Data")
    parser.add_argument("--random_state", "-s", type=int, default=0,
                        help="random state")
    parser.add_argument("--batch_size", "-b", type=int, default=16,
                        help="batch size for fake data")
    parser.add_argument("--num_images_per_classes", "-nipc", type=int, default=1,
                        help="number of images per class")
    parser.add_argument("--output_path", "-o", type=str, default="../data/distillation_data",
                        help="path to save generated images")
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(args.random_state)
    torch.cuda.manual_seed(args.random_state)
    np.random.seed(args.random_state)
    generate_distill_data(
        args.batch_size, args.num_images_per_classes, device, args.output_path)


if __name__ == "__main__":
    main()
