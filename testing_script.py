import torch
from yolox.data.datasets import DAISDataset
from yolox.data import ValTransformYOLinO
from yolox.models import YOLinOHead
from yolox.models import YOLOPAFPN, YOLOX

""" INTERPOLATION AND SAMPLING FOR NUMPY ARRAYS """

"""
def interpolate_line_segment(point1, point2):
    # Calculate the number of steps based on the distance
    distance = np.linalg.norm(np.array(point2) - np.array(point1))
    print("Distances: ")
    print(distance)

    min_points = 2
    density_factor = 10.0

    # Generate interpolated points
    num_points = max(int(np.ceil(distance*density_factor)), min_points) + 1

    # (x2 > x1 and y2 > y1) or (x2 > x1 and y2 < y1)
    if (point2[0] > point1[0] and point2[1] > point1[1]) or (point2[0] > point1[0] and point2[1] < point1[1]):
        x_values = np.linspace(point1[0], point2[0], num_points)
        y_values = np.linspace(point1[1], point2[1], num_points)
        interpolated_points = list(zip(x_values, y_values))

    # (x2 < x1 and y2 < y1) or (x2 < x1 and y2 > y1)
    elif (point2[0] < point1[0] and point2[1] <  point1[1]) or (point2[0] < point1[0] and point2[1] > point1[1]):
        x_values = np.linspace(point2[0], point1[0], num_points)
        y_values = np.linspace(point2[1], point1[1], num_points)
        interpolated_points = list(zip(x_values, y_values))
        interpolated_points.reverse()

    return interpolated_points

# Example usage
point1 = [0.1, 0.3]
point2 = [0.5, 0.4]

# Interpolate without specifying the number of points
interpolated_points_dynamic = interpolate_line_segment(point1, point2)

# Print the interpolated points
print("Interpolated Points (Dynamic):")
print(interpolated_points_dynamic)
print(len(interpolated_points_dynamic))


def sample_points_along_line(interpolated_line, sample_distance=0.03125):
    sampled_points = []

    for i in range(0, len(interpolated_line) - 1):
        point1 = np.array(interpolated_line[i])
        point2 = np.array(interpolated_line[i + 1])

        # Calculate the distance between two consecutive points
        distance = np.linalg.norm(point2 - point1)

        # Calculate the number of steps needed to achieve the sample distance
        num_steps = int(distance / sample_distance)

        # Generate sampled points
        x_values = np.linspace(point1[0], point2[0], num_steps + 1)
        y_values = np.linspace(point1[1], point2[1], num_steps + 1)

        sampled_points.extend(list(zip(x_values, y_values)))

    return sampled_points

sampled_points = sample_points_along_line(interpolated_points_dynamic)
print("Sampled points: ")
print(sampled_points)
print(len(sampled_points))

"""

""" CHECK ONLY DISTANCES BETWEEN START AND END POINTS OF GT AND PREDICTION """
# batch_size = 8
# gt = torch.rand((batch_size, 400, 1, 5))
# pred = torch.rand((batch_size, 400, 1, 5))

# coords_gt = gt[:, :, :, :4].float()
# confs_gt = gt[:, :, :, -1].float()
# confs_gt[:, :20, :] = torch.ones((batch_size, 20, 1))
# confs_gt[:, 20:, :] = torch.zeros((batch_size, 380, 1))

# coords_pred = pred[:, :, :, :4].float()
# confs_pred = pred[:, :, :, -1].float()

# t_conf = 0.5
# t_cell = 0.3

# num_ground_truths = (confs_gt > 0).sum(dim=1)
# print(f"Number of ground truths: {num_ground_truths}")

# num_predictions = (confs_pred > t_conf).sum(dim=1)
# print(f"Number of predictions: {num_predictions}")

# # HIGH LEVEL METRICS
# # True positives
# true_positives = torch.logical_and(confs_gt > 0, confs_pred > t_conf).sum(dim=1)
# print(f"True positives: {true_positives}")

# # False positives
# false_positives = torch.logical_and(confs_gt < 1, confs_pred > t_conf).sum(dim=1)
# print(f"False positives: {false_positives}")

# # True negatives
# true_negatives = torch.logical_and(confs_gt < 1, confs_pred < t_conf).sum(dim=1)
# print(f"True negatives: {true_negatives}")

# # False negatives
# false_negatives = torch.logical_and(confs_gt > 0, confs_pred < t_conf).sum(dim=1)
# print(f"False negatives: {false_negatives}")

# # Precision
# precision = (true_positives / num_predictions).mean(dim=0)
# print(f"Precision: {precision}")

# # Recall
# recall = (true_positives / num_ground_truths).mean(dim=0)
# print(f"Recall: {recall}")

# # F1
# f1_score = (2 * (precision * recall) / (precision + recall)).mean(dim=0)
# print(f"F1 score: {f1_score}")

# # CELL LEVEL METRICS
# calculate_distances = torch.logical_and(confs_gt > 0, confs_pred > t_conf)
# check_distances = torch.logical_and(torch.linalg.norm(coords_gt[:, :, :, :2] - coords_pred[:, :, :, :2], dim=3) < t_cell,
#                                      torch.linalg.norm(coords_gt[:, :, :, 2:] - coords_pred[:, :, :, 2:], dim=3) < t_cell)

# true_positives_cell = torch.logical_and(calculate_distances, check_distances).sum(dim=1)

# print(f"Cell based true positives: {true_positives_cell}")

# # Precision
# precision_cell = (true_positives_cell / num_predictions).mean(dim=0)
# print(f"Cell based precision: {precision_cell}")

# # Recall
# recall_cell = (true_positives_cell / num_ground_truths).mean(dim=0)
# print(f"Cell based recall: {recall_cell}")

######################################################################

data_directory = '/home/manojlovska/Documents/YOLOX/datasets/DAIS-COCO'
batch_size = 8

# train_data = DAISDataset(data_dir=data_directory,
# img_size=(640, 640),
# mag_tape=True,
# name="train",
# preproc=TrainTransformYOLinO(max_labels=1))

valdataset = DAISDataset(data_dir=data_directory,
                         json_file="instances_valid.json",
                         img_size=(640, 640),
                         mag_tape=True,
                         name="valid",
                         preproc=ValTransformYOLinO())

# print(valdataset.__len__())
# print(valdataset[8])
# print(valdataset[0][0].shape)
# print(valdataset.annotations[0][0].shape)

# EVALUATION
test_conf = 0.01
nmsthre = 0.65
num_classes = 1
testdev = False

depth = 0.33
width = 0.50
act = 'relu'

in_channels = [256, 512, 1024]
backbone = YOLOPAFPN(depth,
                     width,
                     in_channels=in_channels,
                     act=act)
head_yolino = YOLinOHead(in_channels=512,
                         num_predictors_per_cell=1,
                         conf=1)  # Only the last layer of features
model = YOLOX(backbone=backbone,
              head=None,
              head_yolino=head_yolino)
model.cuda()

sampler = torch.utils.data.SequentialSampler(valdataset)

dataloader_kwargs = {
    "num_workers": 1,
    "pin_memory": False,
    "sampler": sampler,
}
dataloader_kwargs["batch_size"] = batch_size

# val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
# print(val_loader.batch_size)

# evaluator = DAISEvaluator(
#     dataloader=val_loader,
#     img_size=valdataset.img_size,
#     confthre=test_conf,
#     nmsthre=nmsthre,
#     num_classes=num_classes,
#     testdev=testdev,
#     mag_tape=True,
# )

# evaluated_outputs = evaluator.evaluate(model=model,
#                                        distributed=False,
#                                        half=False,
#                                        trt_file=None,
#                                        decoder=None,
#                                        test_size=valdataset.img_size,
#                                        return_outputs=False)

# print(evaluated_outputs)
# print(len(evaluated_outputs))
# print(evaluated_outputs[0].shape)
# test_size = (640, 640)
# device = "cpu"

# model = Darknet(depth=53)
# in_tensor = torch.rand((1, 3, 640, 640))
# output = model(in_tensor)

# print("Dark3: ", output["dark3"].size())
# print("Dark4: ",output["dark4"].size())
# print("Dark5: ",output["dark5"].size())

# yolo_pafpn = YOLOPAFPN()
# output_fpn = yolo_pafpn(in_tensor)
# print("Output yolo_fpn: ", output_fpn)

# yolino_head = YOLinOHead(in_channels=512, num_predictors_per_cell=1, num_classes=3)
# yolox_head = YOLOXHead(num_classes=3)

# output_head_yolino = yolino_head(output["dark5"])
# print("Output head yolino: ", output_head_yolino)

# yolino_head_fpn = YOLinOHead(in_channels=1024, num_predictors_per_cell=1, num_classes=1)
# output_head_yolino_fpn = yolino_head_fpn(output_fpn[2])
# print("Output head yolino fpn: ", output_head_yolino_fpn)

# output_head_yolox = yolox_head(output_fpn)
# print("Output head yolox: ", output_head_yolox.size())

# darknet_yolino = YOLOX(backbone=yolo_pafpn, head=None, head_yolino=yolino_head)
# output_all = darknet_yolino(in_tensor)


# ############################################################################################
# ########################################## LOSSES ##########################################
# ############################################################################################
# # Parameters
# num_predictors = 1

# # Input
# in_tensor = torch.rand((1, 3, 640, 640))

# # Label - we assume that there is one line in the image with 4 points
# n = 4 # number of points in the line
# label_tensor = torch.rand((1, n, 2))

# # Target tensor
# target_tensor = torch.rand((1, 5, 20, 20))

# # Reshape target tensor
# batch_size = target_tensor.shape[0]
# target_tensor = target_tensor.permute(0, 2, 3, 1)
# target_tensor = target_tensor.reshape(batch_size,
#                                       -1,
#                                       num_predictors,
#                                       5)
# [batch_size, num_cells, num_predictors, coordinates + conf_score]

# # Darknet
# model = Darknet(depth=53)
# output = model(in_tensor)

# # YOLOv3
# yolo_pafpn = YOLOPAFPN()
# output_fpn = yolo_pafpn(in_tensor)

# # YOLinO head
# yolino_head = YOLinOHead(in_channels=512, num_predictors_per_cell=1, num_classes=3)
# output_head_yolino = yolino_head(output["dark5"])

# yolino_head_fpn = YOLinOHead(in_channels=1024, num_predictors_per_cell=1, conf=1)
# output_head_yolino_fpn = yolino_head_fpn(output_fpn[2])

# outputs = output_head_yolino_fpn
# L_loc = []
# L_resp = []
# total_loss = None

# batch_size = outputs.shape[0]
# pred = outputs.permute(0, 2, 3, 1)
# pred = pred.reshape(batch_size,
#                     -1,
#                     num_predictors,
#                     5)
# [batch_size, num_cells, num_predictors, coordinates + conf_score]

# coords = pred[:, :, :, :4]
# conf_scores = pred[:, :, :, -1:]

# p = 0.5
# # For Cartesian points
# for batch_idx in range(batch_size): # for every batch
#     euc_distances = []
#     conf = []
#     for i in range(pred.shape[1]): # for every cell
#         coords_cell_gt = target_tensor[batch_idx, i, :, :4] # Ground truth coordinates
#         coords_cell_gt = coords_cell_gt.numpy()[0]

#         Ground truth confidence score
#         (either 1 or 0, either there is a line segment inside the cell or not)
#         conf_score_cell_gt = target_tensor[batch_idx, i, :, -1:]
#         conf_score_cell_gt = conf_score_cell_gt.item()

#         coords_cell_pred = coords[batch_idx, i, :, :] # Predicted coordinates for the cell
#         coords_cell_pred = coords_cell_pred.detach().numpy()[0]

#         Predicted confidence score for the cell
#         conf_score_cell_pred = conf_scores[batch_idx, i, :, :]
#         conf_score_cell_pred = conf_score_cell_pred.sigmoid().item()

#         # Distance calculation between gt and pred coords in that cell
#         euc_dist = np.linalg.norm(coords_cell_gt[:2] - coords_cell_pred[:2]) + \
#                   np.linalg.norm(coords_cell_gt[2:] - coords_cell_pred[2:])
#         euc_distances.append(euc_dist)

#         # Confidence score error
#         if conf_score_cell_gt:
#             conf.append((conf_score_cell_pred-1)**2)
#         else:
#             conf.append(0.)

#     L_loc = sum(euc_distances) # Do i need to normalize it?
#     L_resp = sum(conf)

#     total_loss = p*L_loc + (1-p)*L_resp

in_tensor = torch.rand((8, 3, 640, 640))
target_tensor = torch.rand((8, 5, 20, 20))

# yolo_pafpn = YOLOPAFPN()
# yolino_head_fpn = YOLinOHead(in_channels=1024, num_predictors_per_cell=1, conf=1)

# model = YOLOX(backbone=yolo_pafpn, head=None, head_yolino=yolino_head_fpn)
# model_predictions = model(in_tensor, target_tensor)
# print(model_predictions)

# target_tensor = torch.rand((1, 5, 20, 20))
# model2 = YOLOX(backbone=yolo_pafpn, head=yolox_head_fpn, head_yolino=None)
# model_predictions2 = model2(in_tensor, target_tensor)
# print(model_predictions2)

############################################################
# TEST TRAINING
# in_channels = [256, 512, 1024]
# backbone = YOLOPAFPN(in_channels=in_channels)
# head = YOLinOHead(in_channels=1024, num_predictors_per_cell=1, conf=1)
# model = YOLOX(backbone, head)
# model.train()
# print(model(in_tensor))

############################################################
# TEST TRAINER
# from .train_yolino import Exp
# exp = Exp()

# trainer = exp.get_trainer(args)
# trainer.train()
