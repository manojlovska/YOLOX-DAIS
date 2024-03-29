import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import CacheDataset, cache_read_img
from .convert2cartesian import Converter
import multiprocessing
from loguru import logger


class DAISDataset(CacheDataset):
    """
    COCO dataset class.
    """

    def __init__(
        self,
        data_dir=None,
        json_file="instances_train.json",
        name="train",
        img_size=(640, 640),
        mag_tape=False,
        preproc=None,
        cache=False,
        cache_type="ram",
    ):
        """
        DAIS dataset initialization.
        Args:
            data_dir (str): dataset root directory
            json_file (str): DAIS json file name
            name (str): DAIS data name (e.g. 'train' or 'valid')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        if data_dir is None:
            data_dir = os.path.join(get_yolox_datadir(), "DAIS-COCO")
        self.data_dir = data_dir
        self.json_file = json_file
        self.mag_tape = mag_tape
        self.square_size = 32
        self.converter = Converter(width=img_size[0],
                                   height=img_size[1],
                                   square_size=self.square_size)

        self.coco = COCO(os.path.join(self.data_dir, "annotations", self.json_file))
        self.ids = self.coco.getImgIds()
        self.num_imgs = len(self.ids)
        self.class_ids = sorted(self.coco.getCatIds())
        self.cats = self.coco.loadCats(self.coco.getCatIds())
        self._classes = tuple([c["name"] for c in self.cats])
        self.name = name
        self.img_size = img_size
        self.preproc = preproc
        self.annotations = self._load_mag_tape_annotations() if self.mag_tape else self._load_coco_annotations()

        path_filename = [os.path.join(name, anno[3]) for anno in self.annotations]

        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=data_dir,
            cache_dir_name=f"cache_{name}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in self.ids]

    def load_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        width = im_ann["width"]
        height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        bbox_annotations = [
            annotations[i] for i in range(len(annotations)) if "bbox" in annotations[i]
            ]

        # Only bounding boxes
        objs = []
        count = 0
        for obj in bbox_annotations:
            x1 = np.max((0, obj["bbox"][0]))
            y1 = np.max((0, obj["bbox"][1]))
            x2 = np.min((width, x1 + np.max((0, obj["bbox"][2]))))
            y2 = np.min((height, y1 + np.max((0, obj["bbox"][3]))))

            if int(obj["attributes"]["occluded"]) != 3:
                if obj["area"] > 0 and x2 >= x1 and y2 >= y1:
                    obj["clean_bbox"] = [x1, y1, x2, y2]
                    objs.append(obj)
            else:
                count += 1
                continue

        num_objs = len(objs)

        res = np.zeros((num_objs, 5))
        for ix, obj in enumerate(objs):
            cls = self.class_ids.index(obj["category_id"])
            res[ix, 0:4] = obj["clean_bbox"]
            res[ix, 4] = cls

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r

        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        if count > 0:
            logger.info("For image {}, {} annotations were excluded".format(file_name, count))

        return (res, img_info, resized_info, file_name)

    def _load_mag_tape_annotations(self):
        pool = multiprocessing.Pool(16)
        return pool.map(self.load_mag_tape_anno_from_ids, self.ids)

    def load_mag_tape_anno_from_ids(self, id_):
        im_ann = self.coco.loadImgs(id_)[0]
        original_width = im_ann["width"]
        original_height = im_ann["height"]
        anno_ids = self.coco.getAnnIds(imgIds=[int(id_)], iscrowd=False)
        annotations = self.coco.loadAnns(anno_ids)

        # Split the annotations to bboxes and magnetic tape
        mag_tape_annotations = [
            annotations[i] for i in range(len(annotations)) if "line" in annotations[i]
            ]

        # Only polylines
        lines = []
        for line in mag_tape_annotations:
            points = line["line"]
            lines.append(points)

        x_scale = self.img_size[0] / original_width
        y_scale = self.img_size[1] / original_height

        # convert polylines to float and optionally scale them
        for polyline in lines:
            for line in polyline:
                line[0] = float(line[0]) * x_scale
                line[1] = float(line[1]) * y_scale

        cartesian_lines = self.converter.to_cartesian(lines)
        cartesian_lines = np.moveaxis(cartesian_lines, -1, 0)

        # Now the annotations are in shape (5, 20, 20), when convertng to tensor add batch size
        # cartesian_lines = np.moveaxis(cartesian_lines, -1, 1)

        cartesian_lines = np.where(np.isnan(cartesian_lines),
                                   np.zeros_like(cartesian_lines),
                                   cartesian_lines)

        img_info = (original_height, original_width)
        resized_info = (int(self.img_size[0]), int(self.img_size[1]))

        file_name = (
            im_ann["file_name"]
            if "file_name" in im_ann
            else "{:012}".format(id_) + ".jpg"
        )

        logger.info("Image {}".format(file_name))

        return (cartesian_lines, img_info, resized_info, file_name)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]

        img_file = os.path.join(self.data_dir, self.name, file_name)

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        id_ = self.ids[index]
        label, origin_image_size, _, _ = self.annotations[index]
        img = self.read_img(index)

        return img, copy.deepcopy(label), origin_image_size, np.array([id_])

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        # logger.info(f"Target shape:{target.shape}, target type: {type(target)}.")
        return img, target, img_info, img_id
