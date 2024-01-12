# def test_augmentation():
import torch
from train_yolino import Exp
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from torchvision import transforms
import os
import torchvision
import cv2
import numpy as np

to_pil_image = transforms.ToPILImage()

torch.set_printoptions(linewidth=200)
exp = Exp()
yolox_outputs_dir = exp.output_dir
os.makedirs(os.path.join(yolox_outputs_dir, "augment_images"), exist_ok=True)
save_dir = os.path.join(yolox_outputs_dir, "augment_images")

# val_dataset = exp.get_eval_dataset()
# val_dataloader = exp.get_eval_loader(batch_size=16, is_distributed=False)

# for cur_iter, (imgs, ann, info_imgs, ids) in enumerate(val_dataloader):
#     print(imgs[0].shape)
#     save_img_name = os.path.join(save_dir, f"val_batch_{cur_iter}.png")
#     torchvision.utils.save_image(imgs, save_img_name)
    

train_dataset = exp.get_dataset()
# import pdb; pdb.set_trace()
train_dataloader = exp.get_data_loader(batch_size=16, is_distributed=False)

for cur_iter, (imgs, ann, info_imgs, ids) in enumerate(train_dataloader):
    print(imgs[0].shape)
    save_img_name = os.path.join(save_dir, f"batch_{cur_iter}.png")

    for idx, img in enumerate(imgs):
        numpy_image = img.numpy()

        # Convert the numpy array to a cv2 image
        cv2_image = np.transpose(numpy_image, (1, 2, 0))
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        save_img_name_cv2 = os.path.join(save_dir, f"batch_{cur_iter}_{idx}.png")
        cv2.imwrite(save_img_name_cv2, cv2_image) 


    
    torchvision.utils.save_image(imgs, save_img_name)
    import pdb; pdb.set_trace()

# import pdb; pdb.set_trace()



