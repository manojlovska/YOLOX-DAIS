# def test_visualize_batch(batch, batch_size, dataset_type):
# 	import matplotlib.pyplot as plt
# 	# initialize a figure
# 	fig = plt.figure("{} batch".format(dataset_type),
# 		figsize=(batch_size, batch_size))
# 	# loop over the batch size
# 	for i in range(0, batch_size):
# 		# create a subplot
# 		ax = plt.subplot(2, 4, i + 1)
# 		# grab the image, convert it from channels first ordering to
# 		# channels last ordering, and scale the raw pixel intensities
# 		# to the range [0, 255]
# 		image = batch[0][i].cpu().numpy()
# 		image = image.transpose((1, 2, 0))
# 		image = (image * 255.0).astype("uint8")
# 		# grab the label id and get the label from the classes list
# 		idx = batch[1][i]
# 		# show the image along with the label
# 		plt.imshow(image)
# 		plt.axis("off")
# 	# show the plot
# 	plt.tight_layout()
# 	plt.show()

# def test_dais():
# 	import torch
# 	from train_yolino import Exp
# 	torch.set_printoptions(linewidth=200)
# 	exp = Exp()
# 	train_dataset = exp.get_dataset()
# 	assert len(train_dataset=0)
	
# OK
# def test_train_inputs():
# 	import torch
# 	from yolox.models.yolino_head import YOLinOHead
# 	from train_yolino import Exp
# 	import matplotlib.pyplot as plt
# 	from torchvision.utils import make_grid
	
# 	torch.set_printoptions(linewidth=200)
# 	exp = Exp()
# 	train_dataset = exp.get_dataset()
# 	train_dataloader = exp.get_data_loader(batch_size=8, is_distributed=False)
# 	import pdb; pdb.set_trace()
# 	for cur_iter, (imgs, ann, info_imgs, ids) in enumerate(train_dataloader):
# 		print(cur_iter)
# 		print(f"Imgs: {imgs}, \n annotations: {ann}, \n info_imgs: {info_imgs}, \n ids: {ids}")


# 	assert len(train_dataloader) > 0
	
def test_val_inputs():
	import torch
	from yolox.models.yolino_head import YOLinOHead
	from train_yolino import Exp
	import matplotlib.pyplot as plt
	from torchvision.utils import make_grid
	
	torch.set_printoptions(linewidth=200)
	exp = Exp()
	val_dataset = exp.get_eval_dataset(batch_size=8, is_distributed=False)
	val_dataloader = exp.get_eval_loader(batch_size=8, is_distributed=False)
	import pdb; pdb.set_trace()
	for val_cur_iter, (val_imgs, val_ann, val_info_imgs, val_ids) in enumerate(val_dataloader):
		print(val_cur_iter)
		print(f"Imgs: {val_imgs}, \n annotations: {val_ann}, \n info_imgs: {val_info_imgs}, \n ids: {val_ids}")
	
	
