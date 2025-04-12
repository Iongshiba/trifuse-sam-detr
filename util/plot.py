import torch

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Rectangle
from PIL import Image
from util.box_ops import box_cxcywh_to_xywh
from util.misc import nested_tensor_from_tensor_list


def plot_bboxes_batch(
    images, predicted_bboxes_batch, ground_truth_bboxes_batch, batch_size
):
    grid_size = int(
        np.ceil(np.sqrt(batch_size))
    )  # Determine the grid dimensions (square-like)

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15), squeeze=False)
    axes = axes.flatten()

    for i in range(batch_size):
        image = images[i]
        predicted_bboxes = predicted_bboxes_batch[i]
        ground_truth_bboxes = ground_truth_bboxes_batch[i]

        ax = axes[i]
        ax.imshow(image)

        for gt_bbox in ground_truth_bboxes:
            x, y, w, h = gt_bbox
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="g", facecolor="none")
            ax.add_patch(rect)

        for pred_bbox in predicted_bboxes:
            x, y, w, h = pred_bbox
            rect = Rectangle((x, y), w, h, linewidth=2, edgecolor="r", facecolor="none")
            ax.add_patch(rect)

        ax.axis("off")  # Turn off axis for better visualization

    for i in range(batch_size, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    return fig


@torch.no_grad()
def plot_img(model, dataset, device):
    model.eval()
    images = []
    pd_boxes = []
    gt_boxes = []

    for i in range(4):
        item = dataset[i]
        sample = nested_tensor_from_tensor_list([item[0].to(device)])
        target = item[1]

        pred = model(sample)

        images.append(Image.open(dataset.get_img_path(i)))
        img_w, img_h = dataset.get_img_size(i)

        gt_box = box_cxcywh_to_xywh(target["boxes"])
        gt_box[:, [0, 1, 2, 3]] *= torch.tensor([img_w, img_h, img_w, img_h])
        gt_boxes.append(gt_box)

        pd_box = box_cxcywh_to_xywh(pred["pred_boxes"].squeeze().detach().cpu())
        pd_box[:, [0, 1, 2, 3]] *= torch.tensor([img_w, img_h, img_w, img_h])
        pd_boxes.append(pd_box)

    stats = {
        "image_with_bboxes": plot_bboxes_batch(
            images,
            pd_boxes,
            gt_boxes,
            4,
        ),
    }

    return stats
