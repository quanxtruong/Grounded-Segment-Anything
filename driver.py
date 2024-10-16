import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pykinect_azure as pykinect
from visualizer import Open3dVisualizer

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import (
    clean_state_dict, 
    get_phrases_from_posmap
)

# Segment Anything
from segment_anything import (
    sam_model_registry,
    sam_hq_model_registry,
    SamPredictor
)

def prepare_image(image: np.ndarray):
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),  # Can switch to T.Resize for consistency
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    image = image[:, :, :3][:, :, [2, 1, 0]]  # Convert to RGB
    image = image.reshape((720, 1280, 3))
    image_pil = Image.fromarray(image)
    image_tensor, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image_tensor

def load_model(model_config_path, model_checkpoint_path, bert_base_uncased_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path

    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)

    model.eval()
    return model

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cpu"):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    model = model.to(device)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]
        boxes = outputs["pred_boxes"].cpu()[0]

    filt_mask = logits.max(dim=1)[0] > box_threshold
    logits_filt = logits[filt_mask]
    boxes_filt = boxes[filt_mask]

    tokenized = model.tokenizer(caption)
    pred_phrases = [
        get_phrases_from_posmap(logit > text_threshold, tokenized, model.tokenizer) +
        (f"({str(logit.max().item())[:4]})" if with_logits else "")
        for logit, _ in zip(logits_filt, boxes_filt)
    ]

    return boxes_filt, pred_phrases

def show_mask(mask, ax, random_color=False):
    color = (
        np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        if random_color else np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    )
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax, label):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    ax.text(x0, y0, label)

if __name__ == "__main__":
    # Configuration paths and thresholds
    GROUNDING_DINO_CONFIG = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT = "groundingdino_swint_ogc.pth"
    SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.25
    DEVICE = "cuda"
    BERT_BASE_UNCASED_PATH = None

    # Kinect configuration
    device_config = pykinect.default_configuration
    device_config.color_format = pykinect.K4A_IMAGE_FORMAT_COLOR_BGRA32
    device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_720P
    device_config.depth_mode = pykinect.K4A_DEPTH_MODE_NFOV_2X2BINNED

    pykinect.initialize_libraries()
    kinect = pykinect.start_device(config=device_config)

    # Load models
    gd_model = load_model(GROUNDING_DINO_CONFIG, GROUNDING_DINO_CHECKPOINT, BERT_BASE_UNCASED_PATH, device=DEVICE)
    sam_model = SamPredictor(sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT).to(DEVICE))
    # open3dvisualizer = Open3dVisualizer()

    # Collection of objects found in the scene
    scene = {}

    while True:
        capture = kinect.update()
        ret_color, color_image = capture.get_color_image()  # (720, 1280, 4)
        # ret_points, points = capture.get_pointcloud()
        if not (ret_color):
            continue

        image_pil, image_tensor = prepare_image(color_image)
        boxes, pred_phrases = get_grounding_output(
            gd_model, image_tensor, "human. chair. table.", BOX_THRESHOLD, TEXT_THRESHOLD, device=DEVICE
        )

        sam_model.set_image(color_image[:, :, :3])

        # Adjust boxes for SAM and visualization
        W, H = image_pil.size
        for i in range(boxes.size(0)):
            boxes[i] *= torch.Tensor([W, H, W, H])
            boxes[i][:2] -= boxes[i][2:] / 2
            boxes[i][2:] += boxes[i][:2]

        transformed_boxes = sam_model.transform.apply_boxes_torch(boxes, color_image.shape[:2]).to(DEVICE)
        masks, _, _ = sam_model.predict_torch(
            point_coords=None, point_labels=None, boxes=transformed_boxes.to(DEVICE), multimask_output=False
        )

        # points.reshape((720, 1280, 3)) ## (92160, 3)
        # points[masks[0]]
        # print(masks.shape) ## (6, 1, 720, 1280), 720 * 1280 = 921600

        plt.figure(figsize=(10, 10))
        plt.axis('off')

        color_image = np.array(image_pil)
        for mask, box, label in zip(masks, boxes, pred_phrases):
            mask = mask[0].cpu().numpy()
            x0, y0 = box[0], box[1]
            pixels = color_image * mask[:, :, None]
            label = label[:label.index('(')]

            if label not in scene:
                scene[label] = []
            scene[label].append(pixels)

        for obj in scene:
            print(obj)
            for pixels in scene[obj]:
                plt.imshow(pixels)
                plt.show()

        # Display masks
        # for mask in masks:
        #     show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)

        # ## Grouding DINO output
        # for box, label in zip(boxes, pred_phrases):
        #     show_box(box.numpy(), plt.gca(), label)
        
