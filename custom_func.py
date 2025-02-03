import numpy as np
import cv2

def crop(image, p1, p2):

    def __doc__():
        return     """
    Crop an image using the specified coordinates.
    


    Args:
        image (np.ndarray): Input image
        p1 (tuple): Starting coordinates (x, y)
        p2 (tuple): Ending coordinates (x, y)

    
    Returns:
        np.ndarray: Cropped image
    """
    
    print("CROP FUNCTION CALLED", p1, p2)
    """
    Crop an image using the specified coordinates.
    


    Args:
        image (np.ndarray): Input image
        p1 (tuple): Starting coordinates (x, y)
        p2 (tuple): Ending coordinates (x, y)

    
    Returns:
        np.ndarray: Cropped image
    """
    # Ensure coordinates are integers
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]



    # Get image dimensions
    height, width = image.shape[:2]
    
    # Validate coordinates
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))
    
    # Ensure x1,y1 is top-left and x2,y2 is bottom-right
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    
    # Perform the crop
    cropped_image = image[y1:y2, x1:x2].copy()
    
    return cropped_image

def draw_rectangle(image, x1, y1, x2, y2, color=(0, 255, 0), thickness=2):
    """
    Draw a rectangle on the image (useful for previewing crop area).
    
    Args:
        image (np.ndarray): Input image
        x1, y1 (int): Top-left corner coordinates
        x2, y2 (int): Bottom-right corner coordinates
        color (tuple): BGR color tuple (default: green)
        thickness (int): Line thickness
    
    Returns:
        np.ndarray: Image with rectangle drawn
    """
    img_copy = image.copy()
    cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

def add_text_overlay(image, text, position=(10, 30), 
                    font=cv2.FONT_HERSHEY_SIMPLEX, 
                    font_scale=1, color=(255, 255, 255), 
                    thickness=2):
    """
    Add text overlay to an image.
    
    Args:
        image (np.ndarray): Input image
        text (str): Text to overlay
        position (tuple): (x, y) coordinates for text
        font: OpenCV font
        font_scale (float): Font scale factor
        color (tuple): BGR color tuple
        thickness (int): Text thickness
    
    Returns:
        np.ndarray: Image with text overlay
    """
    img_copy = image.copy()
    # Add black background for better visibility
    cv2.putText(img_copy, text, position, font, font_scale, 
                (0, 0, 0), thickness + 1)
    # Add text in specified color
    cv2.putText(img_copy, text, position, font, font_scale, 
                color, thickness)
    return img_copy

def create_circular_mask(image, center=None, radius=None):
    """
    Create a circular mask for an image.
    
    Args:
        image (np.ndarray): Input image
        center (tuple): (x, y) coordinates of circle center. If None, uses image center
        radius (int): Radius of circle. If None, uses smaller dimension of image
    
    Returns:
        np.ndarray: Circular masked image
    """
    height, width = image.shape[:2]
    
    if center is None:
        center = (width // 2, height // 2)
    if radius is None:
        radius = min(width, height) // 2
    
    Y, X = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    
    mask = dist_from_center <= radius
    
    masked_img = image.copy()
    if len(image.shape) == 3:
        mask = np.dstack([mask] * 3)
    
    masked_img[~mask] = 0
    
    return masked_img

def mirror_image(image, direction='horizontal'):
    """
    Mirror an image horizontally or vertically.
    
    Args:
        image (np.ndarray): Input image
        direction (str): 'horizontal' or 'vertical'
    
    Returns:
        np.ndarray: Mirrored image
    """
    if direction.lower() == 'horizontal':
        return cv2.flip(image, 1)
    elif direction.lower() == 'vertical':
        return cv2.flip(image, 0)
    else:
        raise ValueError("Direction must be either 'horizontal' or 'vertical'")

def create_vignette(image, strength=1.0):
    """
    Apply a vignette effect to the image.
    
    Args:
        image (np.ndarray): Input image
        strength (float): Vignette effect strength (0.0 to 2.0)
    
    Returns:
        np.ndarray: Image with vignette effect
    """
    height, width = image.shape[:2]
    
    # Create radial gradient
    X = np.linspace(-1, 1, width)
    Y = np.linspace(-1, 1, height)
    x, y = np.meshgrid(X, Y)
    
    # Create vignette mask
    radius = np.sqrt(x**2 + y**2)
    mask = 1 - np.clip(radius * strength, 0, 1)
    
    # Apply mask to image
    mask = np.dstack([mask] * 3) if len(image.shape) == 3 else mask
    vignetted = image * mask
    
    return vignetted.astype(np.uint8)
