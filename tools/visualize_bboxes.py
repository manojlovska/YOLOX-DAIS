import cv2


def visualize_gt(img_annotations, color, result_image, show_img=False):

    if len(img_annotations) != 0:
        flag = False
        for img_annotation in img_annotations:
            left = int(img_annotation["bbox"][0])
            top = int(img_annotation["bbox"][1])
            right = int(img_annotation["bbox"][2])
            bottom = int(img_annotation["bbox"][3])

            result_image = cv2.rectangle(result_image, (left, top), (left+right, top+bottom), color)

    else:
        flag = True

    if show_img:
        cv2.imshow('val_image', result_image)

    return result_image, flag
