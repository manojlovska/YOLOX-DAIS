import torch
from yolox.models.yolino_head import YOLinOHead
from yolox.models import YOLinOHead
from yolox.models import YOLOXHead, YOLOPAFPN, YOLOX
import numpy as np

def test_loss_fn():
    # backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head_yolino = YOLinOHead(in_channels=512, num_predictors_per_cell=1, conf=1) # Only the last layer of features
    outputs = np.zeros([1, 5, 20, 20]).reshape(1, 400, 1, 5)
    gt = np.zeros([1, 5, 20, 20])

    outputs = torch.from_numpy(outputs)
    gt = torch.from_numpy(gt)

    loss, _, _, _ = head_yolino.get_losses(outputs, gt)
    assert loss == 0.0


def test_loss_fn_2():
    # backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head_yolino = YOLinOHead(in_channels=512, num_predictors_per_cell=1, conf=1) # Only the last layer of features
    outputs = np.zeros([1, 5, 20, 20]).reshape(1, 400, 1, 5)
    outputs[0, 0, 0, :4] = np.array([0.1, 0.2, 0.2, 0.5])
    outputs[0, 0, 0, -1] = 0.7

    gt = np.zeros([1, 5, 20, 20])
    gt[0, -1, 0, 0] = 1.0

    outputs = torch.from_numpy(outputs)
    gt = torch.from_numpy(gt)

    L_loc, L_resp, L_noresp, total_loss = head_yolino.get_losses(outputs, gt)
    import pdb
    pdb.set_trace()
    assert L_loc > 0.0

def test_loss_fn_3():
    # backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head_yolino = YOLinOHead(in_channels=512, num_predictors_per_cell=1, conf=1) # Only the last layer of features
    outputs = np.zeros([1, 5, 20, 20]).reshape(1, 400, 1, 5)
    outputs[0, 0, 0, :4] = np.array([0.1, 0.2, 0.2, 0.5])
    outputs[0, 0, 0, -1] = 0.7

    gt = np.zeros([1, 5, 20, 20])
    gt[0, -1, 0, 0] = 1.0

    outputs = torch.from_numpy(outputs)
    gt = torch.from_numpy(gt)

    L_loc, L_resp, L_noresp, total_loss = head_yolino.get_losses(outputs, gt)
    assert L_resp > 0.0

def test_loss_fn_4():
    # backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head_yolino = YOLinOHead(in_channels=512, num_predictors_per_cell=1, conf=1) # Only the last layer of features
    outputs = np.zeros([1, 5, 20, 20]).reshape(1, 400, 1, 5)
    outputs[0, 0, 0, :4] = np.array([0.1, 0.2, 0.2, 0.5])
    outputs[0, 0, 0, -1] = 0.7

    gt = np.zeros([1, 5, 20, 20])
    gt[0, -1, 0, 0] = 1.0

    outputs = torch.from_numpy(outputs)
    gt = torch.from_numpy(gt)

    L_loc, L_resp, L_noresp, total_loss = head_yolino.get_losses(outputs, gt)
    assert L_noresp > 0.0

def test_loss_fn_5():
    # backbone = YOLOPAFPN(depth, width, in_channels=in_channels, act=act)
    head_yolino = YOLinOHead(in_channels=512, num_predictors_per_cell=1, conf=1) # Only the last layer of features
    outputs = np.zeros([1, 5, 20, 20]).reshape(1, 400, 1, 5)
    outputs[0, 0, 0, :4] = np.array([0.1, 0.2, 0.2, 0.5])
    outputs[0, 0, 0, -1] = 0.7

    gt = np.zeros([1, 5, 20, 20])
    gt[0, -1, 0, 0] = 1.0

    outputs = torch.from_numpy(outputs)
    gt = torch.from_numpy(gt)

    L_loc, L_resp, L_noresp, total_loss = head_yolino.get_losses(outputs, gt)
    assert total_loss > 0.0

# def test_evaluation():
#     from yolox.evaluators.dais_evaluator import DAISEvaluator
#     from yolox.data.datasets.dais import DAISDataset
#     from train_yolino import Exp
#     from yolox.models.yolino_head import YOLinOHead
#     from yolox.models.yolox import YOLOX

#     exp = Exp()
#     val_dataset = exp.get_eval_dataset()
#     val_dataloader = exp.get_eval_loader(batch_size=1, is_distributed=False)
#     evaluator = exp.get_evaluator(batch_size=1, is_distributed=False)

#     model = exp.get_model().to("cuda")
#     eval_results = evaluator.evaluate(model=model)
#     print(eval_results)

#     assert len(val_dataloader) > 0
