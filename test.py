
import numpy as np
import pykinect_azure as pykinect
import cv2
import open3d as o3d

pykinect.initialize_libraries()

# Modify camera configuration
device_config = pykinect.default_configuration
device_config.color_resolution = pykinect.K4A_COLOR_RESOLUTION_1080P
device_config.depth_mode = pykinect.K4A_DEPTH_MODE_WFOV_2X2BINNED

# Start device
kinect = pykinect.start_device(config=device_config)

cx = 638.4906616210938
cy = 364.21429443359375
fx = 614.5958251953125
fy = 614.3775634765625

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
K_inv = np.linalg.inv(K)

vis = o3d.visualization.Visualizer()

cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)

## need a function to convert (masked) pixels --> point cloud
## this doesn't give good point clouds
while True:
    capture = kinect.update()
    ret_depth, depth_image = capture.get_depth_image()
    if not (ret_depth):
        continue

    cv2.imshow('Depth Image', depth_image)
    cv2.waitKey(1)
    points = np.zeros((depth_image.shape[0], depth_image.shape[1], 3))
    for u in range(len(depth_image)):
        for v in range(len(depth_image[u])):
            if not depth_image[u, v]: # nothing here
                continue

            uv_homog = np.array([u, v, 1])
            point = depth_image[u, v] * (K_inv @ uv_homog.T)
            points[u, v] = point

    points = points.reshape((512 * 512, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    vis.create_window(window_name="Point cloud", width=800, height=800)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

    