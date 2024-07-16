import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch
from image_segmentation.sam_helpers import show_anns, show_box, show_mask, show_points



CHECKPOINT_PATH = 'sam_vit_b_01ec64.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MODEL_TYPE = "vit_b"

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)

# Load SAM model
predictor = SamPredictor(sam)

input_point = np.array([[310, 200]])
input_label = np.array([1])





# Function to apply segmentation and overlay masks
def segment_and_overlay(image_path, output_folder):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Predict masks
    predictor.set_image(image_rgb)

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Make the masks a little bit thicker
    kernel = np.ones((3, 3), np.uint8)  # Kernel for dilation

    expanded_masks = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        expanded_mask = cv2.dilate(mask.astype(np.uint8), kernel, iterations=3)
        expanded_masks.append(expanded_mask)
        plt.figure(figsize=(12, 9))
        plt.imshow(image)
        show_mask(expanded_mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i} with score {score:.2f}")
        plt.axis('off')
        output_path = output_folder + f"/{os.path.basename(image_path).replace('.png', f'_mask_{i}.png')}"
        plt.savefig(output_path, bbox_inches='tight')

    #Take the one with the highest score and save the expanded mask to npy
    best_mask = expanded_masks[np.argmax(scores)]
    print(best_mask.shape)
    np.save(output_folder + f"/{os.path.basename(image_path).replace('.png', '_mask.npy')}", best_mask)

    



# Main function to process all images in a folder
def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    i = 0
    for filename in os.listdir(input_folder):
        i += 1
        if i == 2:
            break
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            segment_and_overlay(image_path, output_folder)
            print(f"Processed {filename}")

# Define input and output folders
input_folder = './pingu/images'
output_folder = './pingu/segmented_images'

# Process all images
process_folder(input_folder, output_folder)
