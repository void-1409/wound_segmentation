import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os

def load_image(path_img):
    """Load an image from the specified path."""
    img = cv2.imread(path_img)
    H, W, _ = img.shape
    return img, H, W

def load_model(path_model):
    """Load the YOLO model from the specified path."""
    return YOLO(path_model)

def segment_image(model, img):
    """Segment the image using the YOLO model."""
    results = model(img)
    return results

def create_masks(results, H, W):
    """Create masks for wound and reference areas from the segmentation results."""
    mask_wound = np.zeros((H, W), dtype=np.uint8)
    mask_reference = np.zeros((H, W), dtype=np.uint8)
    wound_box = None

    if not results:  # Check if results is empty
        return mask_wound, mask_reference, wound_box, False
    
    for result in results:
        if result.masks is None:  # Check if result.masks is None
            return mask_wound, mask_reference, wound_box, False
        
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
    
    # Check if both wound and reference masks are present
    valid_masks = np.any(mask_wound) and np.any(mask_reference)
    
    return mask_wound, mask_reference, wound_box, valid_masks

def calculate_area(mask_wound, mask_reference, area_reference):
    """Calculate the actual wound area using the reference area."""
    pixel_area_wound = np.sum(mask_wound) / 255
    pixel_area_reference = np.sum(mask_reference) / 255

    # Ensure pixel_area_wound and pixel_area_reference are floats
    pixel_area_wound = float(pixel_area_wound)
    pixel_area_reference = float(pixel_area_reference)
    
    if pixel_area_reference > 0:
        converted_area = pixel_area_wound * area_reference / pixel_area_reference
        return round(converted_area, 3)
    else:
        raise ValueError("No reference detected!")

def annotate_image(img, mask_wound, wound_box, actual_wound_area):
    """Annotate the image with the bounding box, wound area, and segmented mask."""
    if wound_box is not None:
        # Draw bounding box around the wound
        x1, y1, x2, y2 = wound_box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Write the wound area on the image
        text = f"Wound Area: {actual_wound_area} cm2"
        cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    # Create a colored mask for the wound
    colored_mask = np.zeros_like(img)
    colored_mask[:, :, 1] = mask_wound  # Use the green channel for the mask
    
    # Blend the original image with the colored mask
    alpha = 0.5  # Transparency factor
    img = cv2.addWeighted(img, 1, colored_mask, alpha, 0)
    
    return img

def save_image(img, output_path):
    """Save the annotated image to the specified path."""
    cv2.imwrite(output_path, img)

def main(input_folder, path_model, area_reference, output_folder):
    model = load_model(path_model)
    
    # List all image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for image_file in image_files:
        # Construct full image path
        path_img = os.path.join(input_folder, image_file)
        
        img, H, W = load_image(path_img)
        results = segment_image(model, img)
        mask_wound, mask_reference, wound_box, valid_masks = create_masks(results, H, W)
        
        if not valid_masks:
            print(f"{image_file}: Skipping, no reference detected!")
            continue
        
        try:
            actual_wound_area = calculate_area(mask_wound, mask_reference, area_reference)
            print(f"{image_file}: Actual wound area: {actual_wound_area} cm²")
            
            # Annotate the image
            annotated_img = annotate_image(img, mask_wound, wound_box, actual_wound_area)
            
            # Construct output path
            output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_annotated.jpg")
            
            # Save the annotated image
            save_image(annotated_img, output_path)
            print(f"Annotated image saved to {output_path}")
            
        except ValueError as e:
            print(f"{image_file}: {e}")

# Parameters
input_folder = "./images/test/"
path_model = "segment_v3.pt"
area_reference = 9.9225  # Known area of the reference object in square centimeters
output_folder = "./images/output/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Run the main function
if __name__ == "__main__":
    main(input_folder, path_model, area_reference, output_folder)