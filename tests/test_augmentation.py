def test_augmentation():
    import torch
    from train_yolino import Exp
    import matplotlib.pyplot as plt
    from torchvision.utils import make_grid
    from torchvision import transforms

    to_pil_image = transforms.ToPILImage()

    torch.set_printoptions(linewidth=200)
    exp = Exp()
    train_dataset = exp.get_dataset()
    train_dataloader = exp.get_data_loader(batch_size=8, is_distributed=False)

    for cur_iter, (imgs, ann, info_imgs, ids) in enumerate(train_dataloader):
        print(imgs[0].shape)
        for img in imgs:
            pil_img = to_pil_image(img)
            pil_img.show()
        import pdb; pdb.set_trace()




