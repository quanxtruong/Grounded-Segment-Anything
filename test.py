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

# cx = 957.9860229492188
# cy = 546.5714111328125
# fx = 921.8936767578125
# fy = 921.5663452148438

K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]) ## takes us from 3d point to 2d pixel coords
K_inv = np.eye(4)
K_inv[:3, :3] = K
K_inv = np.linalg.inv(K_inv) ## takes us from 2d pixel coords to 3d point

k1 = 0.4219205677509308
k2 = -2.3591108322143555
k3 = 1.3102449178695679
k4 = 0.3047582805156708
k5 = -2.1945600509643555
k6 = 1.2463057041168213

p2 = -0.0004227574390824884
p1 = 0.0006175598246045411

D = np.array([k1, k2, p1, p2, k3, k4, k5, k6])

map1, map2 = np.array([]), np.array([])
map1, map2 = cv2.initUndistortRectifyMap(K, D, np.array([]), K, (1280, 720), cv2.CV_32FC1, map1, map2)

def undistortImage(in_mat):
    out = np.array([])
    out = cv2.remap(in_mat, map1, map2, cv2.INTER_NEAREST, out)
    return out

vis = o3d.visualization.Visualizer()

cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)

## need a function to convert (masked) pixels --> point cloud
while True:
    capture = kinect.update()
    ret_depth, depth_image = capture.get_depth_image()
    if not (ret_depth):
        continue

    depth_image = undistortImage(depth_image)
    depth_image = depth_image.astype(np.float32)

    cv2.imshow('Depth Image', depth_image)
    cv2.waitKey(1)

    ## Processing each pixel
    rows, cols = depth_image.shape
    pointcloud = np.zeros((rows, cols, 3))
    for u in range(0, cols, 3):
        for v in range(0, rows, 3):
            depth_value = depth_image[v, u] * 0.001
            if depth_value == 0.0:
                depth_value = 8.

            uv_h = np.array([u, v, 1., 1 / depth_value])
            point = depth_value * (K_inv @ uv_h.T)
            if (np.isnan(point[0]) or np.isnan(point[1]) or np.isnan(point[2])):
                continue

            pointcloud[v, u] = point[:3]

    pointcloud = pointcloud.reshape((rows * cols, 3))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)

    vis.create_window(window_name="Point cloud", width=800, height=800)
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


