 import albumentations as A
import numpy as np
from PIL import Image
import os
from tqdm import tqdm

input_dir = "./2022-seg/data/annotation_JPG_process/"
target_dir = './2022-seg/data/annotation_masked_edited_256'
new_image_path = './2022-seg/data/annotation_JPG_process_edited_256_aug'
new_mask_path = './2022-seg/data/annotation_masked_edited_256_aug'

input_img_paths = sorted([os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith(".JPG")])
target_img_paths = sorted([os.path.join(target_dir, fname) for fname in os.listdir(target_dir) if fname.endswith(".npy") and not fname.startswith(".")])

print(len(target_img_paths))
keys = [os.path.splitext(os.path.basename(path))[0] for path in target_img_paths]

transforms = A.Compose([
    A.RandomBrightnessContrast(p=1.0),
    A.ShiftScaleRotate(p=1.0),
    A.GridDistortion(p=1.0),
    A.ElasticTransform(p=1.0),
    A.HorizontalFlip(p=1.0),
    A.VerticalFlip(p=1.0),
    A.RandomRotate90(p=1.0),
    A.Transpose(p=1.0),
    A.RandomCrop(p=1.0, height=256, width=256),
])


image_size = (256, 256)

for key in tqdm(keys, desc="Processing"):
    image_path = os.path.join(input_dir, f"{key}.JPG")
    mask_path = os.path.join(target_dir, f"{key}.npy")

    image = Image.open(image_path).resize(image_size)
    mask = np.load(mask_path)
    image = np.array(image)
    mask = np.array(mask)

    for index, transform in enumerate(transforms.transforms):
        new_fname = f"{key}_{index}"
        transformed = transform(image=image, mask=mask)

        new_image = transformed["image"]
        new_mask = transformed["mask"]

        new_image = Image.fromarray(new_image.astype(np.uint8))
        new_image.save(os.path.join(new_image_path, f"{new_fname}.JPG"))
        np.save(os.path.join(new_mask_path, f"{new_fname}.npy"), new_mask)