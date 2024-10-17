import numpy as np
import cv2
from PIL import Image
import open3d as o3d
import matplotlib.pyplot as plt

def read_images_pil(color_image_path, depth_image_path):
    color_image = Image.open(color_image_path)
    color_array = np.array(color_image)

    depth_image = Image.open(depth_image_path)
    depth_array = np.array(depth_image)

    if depth_array.dtype == np.uint16:
        depth_array = depth_array.astype(np.float32) / 1000.0  # Assuming depth is in millimeters

    return color_array, depth_array

def calculate_transformation_matrix(pos1, pos2, rot1, rot2):
    rot1_rad = np.radians(rot1)
    rot2_rad = np.radians(rot2)
    
    R1 = np.array([
        [np.cos(rot1_rad[2]), -np.sin(rot1_rad[2]), 0],
        [np.sin(rot1_rad[2]), np.cos(rot1_rad[2]), 0],
        [0, 0, 1]
    ])
    
    R2 = np.array([
        [np.cos(rot2_rad[2]), -np.sin(rot2_rad[2]), 0],
        [np.sin(rot2_rad[2]), np.cos(rot2_rad[2]), 0],
        [0, 0, 1]
    ])
    
    R_relative = np.dot(R2, np.linalg.inv(R1))
    t_relative = np.array(pos2) - np.dot(R_relative, np.array(pos1))
    
    transform = np.eye(4)
    transform[:3, :3] = R_relative
    transform[:3, 3] = t_relative
    
    return transform

def transform_point_cloud(points, transform_matrix):
    homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
    transformed_points = np.dot(homogeneous_points, transform_matrix.T)
    return transformed_points[:, :3]

def combine_and_visualize_point_clouds(pointcloud1, pointcloud2):
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(pointcloud1)
    pcd1.paint_uniform_color([1, 0, 0])  # Red for the first point cloud
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(pointcloud2)
    pcd2.paint_uniform_color([0, 0, 1])  # Blue for the second point cloud
    
    combined_pcd = pcd1 + pcd2
    
    o3d.visualization.draw_geometries([combined_pcd])

def generate_pointcloud(color_image, depth_image, mask, K_inv):
    rows, cols = depth_image.shape
    pointcloud = []
    colors = []

    for v in range(rows):
        for u in range(cols):
            if not mask[v, u]:
                continue

            depth_value = depth_image[v, u]
            if depth_value == 0.0:
                continue

            uv_h = np.array([u, v, 1., 1 / depth_value])
            point = depth_value * (K_inv @ uv_h.T)[:3]
            if np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2]):
                continue

            pointcloud.append(point)
            colors.append(color_image[v, u] / 255.0)

    return np.array(pointcloud), np.array(colors)

def process_image(color_image, depth_image, mask, K_inv):
    # Apply mask to color and depth images
    masked_color = color_image * mask[:, :, None]
    masked_depth = depth_image * mask

    # Generate point cloud
    pointcloud, colors = generate_pointcloud(masked_color, masked_depth, mask, K_inv)

    return pointcloud, colors, masked_color, masked_depth

if __name__ == "__main__":
    # Camera intrinsic parameters (adjust these based on your camera)
    cx, cy = 638.4906616210938, 364.21429443359375
    fx, fy = 614.5958251953125, 614.3775634765625
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K_inv = np.linalg.inv(np.eye(4))
    K_inv[:3, :3] = np.linalg.inv(K)

    # Read images
    color_image_path1 = "images/color_image1.png"
    depth_image_path1 = "images/depth_image1.png"
    mask_path1 = "images/mask_image1.png"
    color_image_path2 = "images/color_image2.png"
    depth_image_path2 = "images/depth_image2.png"
    mask_path2 = "images/mask_image2.png"

    color_image1, depth_image1 = read_images_pil(color_image_path1, depth_image_path1)
    color_image2, depth_image2 = read_images_pil(color_image_path2, depth_image_path2)
    mask1 = np.array(Image.open(mask_path1)) > 0
    mask2 = np.array(Image.open(mask_path2)) > 0

    # Ensure color and depth images have the same dimensions
    color_image1 = cv2.resize(color_image1, (depth_image1.shape[1], depth_image1.shape[0]))
    color_image2 = cv2.resize(color_image2, (depth_image2.shape[1], depth_image2.shape[0]))
    mask1 = cv2.resize(mask1.astype(np.uint8), (depth_image1.shape[1], depth_image1.shape[0])) > 0
    mask2 = cv2.resize(mask2.astype(np.uint8), (depth_image2.shape[1], depth_image2.shape[0])) > 0

    # Process images
    pointcloud1, colors1, masked_color1, masked_depth1 = process_image(color_image1, depth_image1, mask1, K_inv)
    pointcloud2, colors2, masked_color2, masked_depth2 = process_image(color_image2, depth_image2, mask2, K_inv)

    # Visualize masked images
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(masked_color1)
    axes[0, 0].set_title("Masked Color Image 1")
    axes[0, 1].imshow(masked_depth1)
    axes[0, 1].set_title("Masked Depth Image 1")
    axes[1, 0].imshow(masked_color2)
    axes[1, 0].set_title("Masked Color Image 2")
    axes[1, 1].imshow(masked_depth2)
    axes[1, 1].set_title("Masked Depth Image 2")
    plt.tight_layout()
    plt.show()

    # Calculate transformation matrix
    # Assuming the camera moved forward by 67 inches and rotated 180 degrees
    pos1 = (0, 0, 0)
    pos2 = (0, 0, 0)  # Convert 67 inches to meters, negative because we're moving forward 67
    rot1 = (0, 0, 0)
    rot2 = (0, 0, 0)
    transformation_matrix = calculate_transformation_matrix(pos1, pos2, rot1, rot2)

    # Transform pointcloud2
    transformed_pointcloud2 = transform_point_cloud(pointcloud2, transformation_matrix)

    # Combine point clouds
    combined_pointcloud = np.vstack((pointcloud1, transformed_pointcloud2))
    combined_colors = np.vstack((colors1, colors2))

    # Visualize combined point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(combined_pointcloud)
    pcd.colors = o3d.utility.Vector3dVector(combined_colors)
    o3d.visualization.draw_geometries([pcd])

# Note on coordinate system:
# X-axis: Positive to the right in the image plane
# Y-axis: Positive down in the image plane
# Z-axis: Positive into the image (away from the camera)