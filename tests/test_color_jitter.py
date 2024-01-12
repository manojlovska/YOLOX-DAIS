import torch
import torchvision
import glob
import os
from PIL import Image
import torchvision.transforms.functional as F
import torchvision
import numpy as np
import cv2

to_tensor = F.pil_to_tensor
normalize = F.normalize

color_jitter = torchvision.transforms.ColorJitter(brightness=(0,0.5), saturation=(0,0.5), hue=(0, 0.5), contrast=(0, 0.5))

images_dir = "/home/manojlovska/Documents/Projects/YOLOX-DAIS/datasets/DAIS-COCO/train/naloga_1"
image_filenames = sorted(glob.glob(os.path.join(images_dir, "*.jpg"), recursive=True))
save_dir = "/home/manojlovska/Documents/Projects/YOLOX-DAIS/YOLOX_outputs/test_color_jitter"
os.makedirs(save_dir, exist_ok=True)

test_images = image_filenames

for image_name in test_images:
    img = Image.open(image_name)
    img_tensor = to_tensor(img).unsqueeze(0).to(torch.float32)
    # normalized_tensor = normalize(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transformed_tensor = color_jitter(img_tensor)

    save_image_name = os.path.join(save_dir,"norm_" + os.path.basename(image_name))
    torchvision.utils.save_image(transformed_tensor, save_image_name)

    save_image_name_cv2 = os.path.join(save_dir,"not_norm_cv2_" + os.path.basename(image_name))
    numpy_image = transformed_tensor.squeeze(0).numpy()
    import pdb; pdb.set_trace()

    # Convert the numpy array to a cv2 image
    cv2_image = np.transpose(numpy_image, (1, 2, 0))
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(save_image_name_cv2, cv2_image) 







