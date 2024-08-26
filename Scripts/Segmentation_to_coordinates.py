import numpy as np
import cv2
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

def visualize(**images):
    """
    Plot multiple images in one row.

    Arguments:
        **images: Key-value pairs where the key is the title of the image and the value is the image array.
    """
    n_images = len(images)
    plt.figure(figsize=(20, 8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]) 
        plt.yticks([])
        # Set title from the parameter names
        plt.title(name.replace('_', ' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format.
    
    Arguments:
        label: The 2D array segmentation image label.
        label_values: List of values corresponding to each class.
        
    Returns:
        A 3D array with the same width and height as the input, but with a depth size of num_classes.
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

def reverse_one_hot(image):
    """
    Transform a 3D one-hot encoded array into a 2D array with class keys.
    
    Arguments:
        image: The one-hot format image (3D array).
        
    Returns:
        A 2D array with the same width and height as the input, but with a single channel where each pixel value is the classified class key.
    """
    x = np.argmax(image, axis=-1)
    return x

def colour_code_segmentation(image, label_values):
    """
    Colour code the segmentation results based on class keys.
    
    Arguments:
        image: A 2D array where each value represents the class key.
        label_values: List of values assigned to each position in the one-hot encoded vector.
        
    Returns:
        Colour coded image for segmentation visualization.
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x

def mask_to_polygons(mask):
    """
    Convert an image mask into polygons. Returns two lists:
    - List of shapely polygons (not normalized)
    - List of normalized shapely polygons (coordinates between 0 and 1)

    Arguments:
        mask: A binary mask image.
        
    Returns:
        Two lists of shapely Polygon objects: 
        1. List of non-normalized polygons
        2. List of normalized polygons
    """
    mask = mask.astype(bool)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    polygons = []
    normalized_polygons = []
    for contour in contours:
        try:
            polygon = contour.reshape(-1, 2).tolist()
            
            # Normalize coordinates between 0 and 1
            normalized_polygon = [[round(coord[0] / mask.shape[1], 4), round(coord[1] / mask.shape[0], 4)] for coord in polygon]
        
            # Convert to shapely polygon (non-normalized)
            polygon_shapely = Polygon(polygon)
            simplified_polygon = polygon_shapely.simplify(0.85, preserve_topology=True)
            polygons.append(simplified_polygon)

            # Add normalized polygon
            normalized_polygons.append(Polygon(normalized_polygon))
        except Exception as e:
            pass

    return polygons, normalized_polygons

def plot_polygons(polygons, title):
    """
    Plot polygons on the mask image.

    Arguments:
        polygons: List of shapely Polygon objects to be plotted.
        title: Title of the plot.
        
    Returns:
        List of coordinates for the plotted polygons.
    """
    fig, ax = plt.subplots()
    ax.imshow(mask, cmap='gray')
    coordinates = []
    for polygon in polygons:
        if polygon.is_empty:
            continue
        exterior_coords = np.array(polygon.exterior.coords)
        coordinates.append(exterior_coords)
        patch = plt.Polygon(exterior_coords, closed=True, fill='red', edgecolor='red', linewidth=2)
        ax.add_patch(patch)
    
    ax.set_title(title)
    plt.show()
    return coordinates
