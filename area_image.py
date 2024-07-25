import cv2
import numpy as np
from ultralytics import YOLO
import torch

def load_image(path_img):
    # load image from given path
    img = cv2.imread(path_img)
    H, W, _ = img.shape
    return img, H, W

def load_model(path_model):
    # load yolov8 segmentation model from given path
    return YOLO(path_model)

def segment_image(model, img):
    # runs segmentation model on input image
    results = model(img)
    return results

def create_masks(results, H, W):
    # Create masks of wound and reference for calculation
    mask_wound = np.zeros((H, W), dtype=np.uint8)
    mask_reference = np.zeros((H, W), dtype=np.uint8)
    wound_box = None
    
    for result in results:
        masks = result.masks.data
        boxes = result.boxes.data.cpu().numpy()
        clss = boxes[:, 5]
        
        indices_wound = np.where(clss == 1)
        indices_reference = np.where(clss == 0)
        
        if indices_wound[0].size > 0:
            masks_wound = masks[indices_wound]
            mask_wound = torch.any(masks_wound, dim=0).int().cpu().numpy() * 255
            # Get the first bounding box for the wound
            wound_box = boxes[indices_wound][0][:4].astype(int)
        
        if indices_reference[0].size > 0:
            masks_reference = masks[indices_reference]
            mask_reference = torch.any(masks_reference, dim=0).int().cpu().numpy() * 255
    
    return mask_wound, mask_reference, wound_box

def calculate_area(mask_wound, mask_reference, area_reference):
    # Calculate the real area of wound using reference
    pixel_area_wound = np.sum(mask_wound) / 255
    pixel_area_reference = np.sum(mask_reference) / 255
    
    if pixel_area_reference > 0:
        converted_area = pixel_area_wound * area_reference / pixel_area_reference
        return round(converted_area, 3)
    else:
        raise ValueError("No reference detected!")

def annotate_image(img, mask_wound, wound_box, actual_wound_area):
    # annotate the image with bounding box and wound mask
    if wound_box is not None:
        # Draw bounding box around the wound
        x1, y1, x2, y2 = wound_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Write the wound area on the image
        text = f"Wound Area: {actual_wound_area} cm2"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Create a colored mask for the wound
    colored_mask = np.zeros_like(img)
    colored_mask[:, :, 1] = mask_wound  # using the green channel

    alpha = 0.5
    img = cv2.addWeighted(img, 1, colored_mask, alpha, 0)

    return img

def save_image(img, output_path):
    # save the image to given path
    cv2.imwrite(output_path, img)

def main(path_img, path_model, area_reference, output_path):
    img, H, W = load_image(path_img)
    model = load_model(path_model)
    results = segment_image(model, img)
    mask_wound, mask_reference, wound_box = create_masks(results, H, W)
    
    try:
        actual_wound_area = calculate_area(mask_wound, mask_reference, area_reference)
        print(f"Actual wound area: {actual_wound_area} cm2")
        
        # Annotate the image
        annotated_img = annotate_image(img, mask_wound, wound_box, actual_wound_area)
        
        # Save the annotated image
        save_image(annotated_img, output_path)
        print(f"Annotated image saved to {output_path}")
        
    except ValueError as e:
        print(e)

# Parameters
path_img = "./images/test/JOSE EMILIOIMG1435.jpg"
path_model = "segment_v3.pt"
area_reference = 9.9225  # Known area of the reference object in square centimeters
output_path = "./images/test/JOSE_EMILIOIMG1435_annotated.jpg"  # Output path for the annotated image

# Run the main function
if __name__ == "__main__":
    main(path_img, path_model, area_reference, output_path)