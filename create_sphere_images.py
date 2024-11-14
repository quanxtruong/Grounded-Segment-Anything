import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

def create_sphere_depth_map(size=256, distance=2.0):
    # Create coordinate grids
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Calculate radius at each point
    R = np.sqrt(X**2 + Y**2)
    
    # Create sphere depth map
    depth_map = np.full((size, size), 255)  # Initialize with background depth (white)
    
    # Create mask for sphere
    mask = R <= 1
    
    # Calculate Z coordinates of sphere surface
    Z = np.zeros((size, size))
    Z[mask] = np.sqrt(1 - X[mask]**2 - Y[mask]**2)
    
    # Set depth values (distance from camera)
    depth_map[mask] = distance + Z[mask]
    
    # Create border mask (ring around the sphere)
    border_thickness = 2  # Adjust this value to change border thickness
    border_mask = (R <= 1 + border_thickness/size) & (R > 1)
    
    # Normalize depth values to 0-255 range for visualization
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_map_normalized = ((depth_map - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
    
    # Add black border
    depth_map_normalized[border_mask] = 0  # Set border pixels to black
    
    return depth_map_normalized

# Create front and back views
size = 256
front_distance = 2.0
back_distance = 5.0

front_view = create_sphere_depth_map(size, front_distance)
back_view = create_sphere_depth_map(size, back_distance)

# Save images
# ImageOps.invert(Image.fromarray(front_view)).save('sphere_mask_front.png')
# ImageOps.invert(Image.fromarray(back_view)).save('sphere_mask_back.png')

Image.fromarray(front_view).save('sphere_color_front.png')
Image.fromarray(back_view).save('sphere_color_back.png')

# Display images
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(front_view, cmap='gray')
plt.title(f'Front View (Distance: {front_distance}m)')
plt.axis('off')

plt.subplot(122)
plt.imshow(back_view, cmap='gray')
plt.title(f'Back View (Distance: {back_distance}m)')
plt.axis('off')

plt.tight_layout()
plt.show()