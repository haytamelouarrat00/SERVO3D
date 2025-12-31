import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import open3d as o3d
from scipy.spatial.transform import Rotation as R

class Camera:
    def __init__(self, pose, width=1280, height=720, fov_deg=60.0):
        self.width = width
        self.height = height
        self.fov_deg = fov_deg
        self.intrinsics = o3d.camera.PinholeCameraIntrinsic()
        fov_rad = np.deg2rad(self.fov_deg)
        self.fy = (height / 2.0) / np.tan(fov_rad / 2.0)
        self.fx = self.fy
        self.cx, self.cy = width / 2.0 - 0.5, height / 2.0 - 0.5
        self.intrinsics.set_intrinsics(
            self.width, self.height, self.fx, self.fy, self.cx, self.cy
        )
        self.update_pose(pose)

    def update_pose(self, pose):
        """Updates the internal pose state without creating a new object."""
        self.pose = np.array(pose, dtype=np.float64)
    
    def get_view_vectors(self):
        """Converts 1x6 pose to eye, lookat, and up vectors used by the renderer"""
        x, y, z, rx, ry, rz = self.pose

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        R = R_z @ R_y @ R_x

        eye = np.array([x, y, z])
        forward = R @ np.array([0, 0, -1])
        lookat = eye + forward
        up = R @ np.array([0, 1, 0])

        return eye, lookat, up

    def get_view(self, render):
        """Returns the current pose as a 1x6 array."""
        eye, lookat, up = self.get_view_vectors()
        render.setup_camera(60.0, lookat, eye, up)
        return np.asarray(render.render_to_image())
    
    def capture_depth(self, renderer):
        """
        Applies the current camera pose to the renderer and returns a depth map.
        """
        eye, lookat, up = self.get_view_vectors()
        renderer.setup_camera(60.0, lookat, eye, up)

        # Render depth
        # z_in_view_space=True means the pixel values represent true distance (meters).
        # z_in_view_space=False means raw depth buffer (0.0 to 1.0 non-linear).
        return renderer.render_to_depth_image(z_in_view_space=True)

    def visualize_depth(self, depth_map, colormap='viridis', min_depth=None, max_depth=None, save_path=None):
        """
        Visualizes a depth map with a colormap.

        Args:
            depth_map: 2D numpy array of depth values (in meters)
            colormap: Matplotlib colormap name (e.g., 'viridis', 'plasma', 'inferno', 'jet')
            min_depth: Minimum depth for normalization (auto if None)
            max_depth: Maximum depth for normalization (auto if None)
            save_path: If provided, saves the visualization to this path

        Returns:
            RGB image as numpy array (H, W, 3) with values in [0, 255]
        """
        depth = np.asarray(depth_map)

        # Handle infinite values (background/sky)
        valid_mask = np.isfinite(depth)
        if not np.any(valid_mask):
            # All values are infinite, return black image
            return np.zeros((*depth.shape, 3), dtype=np.uint8)

        # Determine depth range
        if min_depth is None:
            min_depth = np.min(depth[valid_mask])
        if max_depth is None:
            max_depth = np.max(depth[valid_mask])

        # Normalize depth to [0, 1]
        if max_depth > min_depth:
            normalized = (depth - min_depth) / (max_depth - min_depth)
        else:
            normalized = np.zeros_like(depth)

        # Clamp values
        normalized = np.clip(normalized, 0, 1)

        # Set invalid pixels to 0 (will appear as darkest color)
        normalized[~valid_mask] = 0

        # Apply colormap
        cmap = cm.get_cmap(colormap)
        colored = cmap(normalized)[:, :, :3]  # Remove alpha channel

        # Convert to uint8
        rgb_image = (colored * 255).astype(np.uint8)

        if save_path:
            plt.imsave(save_path, rgb_image)

        return rgb_image
    
    def project(self, point_cam):
        """
        Projects a 3D point in camera coordinates to 2D pixel coordinates.

        Args:
            point_cam: 3D point as a numpy array [X, Y, Z] in camera coordinates
        Returns:
            2D pixel coordinates as a numpy array [u, v]
        """
        X, Y, Z = point_cam
        if Z <= 0:
            return None  # Point is behind the camera

        u = (self.fx * X) / Z + self.cx
        v = (self.fy * Y) / Z + self.cy

        return np.array([u, v])
    
    def backproject(self, pixel, depth):
        """
        Backprojects a 2D pixel with depth to a 3D point in camera coordinates.

        Args:
            pixel: 2D pixel coordinates as a numpy array [u, v]
            depth: Depth value (Z) at the pixel
        Returns:
            3D point as a numpy array [X, Y, Z] in camera coordinates
        """
        u, v = pixel
        Z = depth
        X = (u - self.cx) * Z / self.fx
        Y = (v - self.cy) * Z / self.fy

        return np.array([X, Y, Z])
    
    def get_rotation_matrix(self):
        roll, pitch, yaw = self.pose[3:]
        return R.from_euler("xyz", [roll, pitch, yaw], degrees=True).as_matrix()
    
    def transform_frame(self, point, R, t):
        """Transforms a 3D point using rotation R and translation t."""
        return R @ point + t
    
    