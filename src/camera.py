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

    def visualize_depth(
        self,
        depth_map,
        colormap="viridis",
        min_depth=None,
        max_depth=None,
        save_path=None,
    ):
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
        """
        Get rotation matrix from current pose.
        Angles are in radians using XYZ Euler convention.

        Returns:
            R: 3x3 rotation matrix
        """
        rx, ry, rz = self.pose[3:]

        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)

        R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        return R_z @ R_y @ R_x

    def transform_frame(self, point, R, t):
        """Transforms a 3D point using rotation R and translation t."""
        return R @ point + t

    def backproject_points(self, keypoints, depth_map):
        """
        Backprojects keypoints to 3D using depth map.
        Returns both 3D points and corresponding 2D pixel coords (for IBVS).

        Args:
            keypoints: List of cv2.KeyPoint objects
            depth_map: Depth map array

        Returns:
            points_3d: Nx3 array of 3D points in camera frame
            pixels_2d: Nx2 array of 2D pixel coordinates
        """
        depth_arr = np.asarray(depth_map)
        points_3d = []
        pixels_2d = []

        for kp in keypoints:
            u, v = kp.pt
            u_int, v_int = int(round(u)), int(round(v))

            if not (
                0 <= v_int < depth_arr.shape[0] and 0 <= u_int < depth_arr.shape[1]
            ):
                raise Exception("Keypoint out of depth map bounds")

            z = depth_arr[v_int, u_int]

            if not np.isfinite(z) or z <= 0:
                raise Exception("Invalid depth at keypoint location")

            point_3d = self.backproject(np.array([u, v]), z)
            points_3d.append(point_3d)
            pixels_2d.append([u, v])

        return np.array(points_3d), np.array(pixels_2d)

    @staticmethod
    def rotation_matrix_to_euler(R):
        """
        Convert rotation matrix to Euler angles (XYZ convention).
        Inverse of get_rotation_matrix_from_euler.

        This extracts Euler angles assuming the rotation matrix was constructed as:
        R = Rz @ Ry @ Rx

        Args:
            R: 3x3 rotation matrix

        Returns:
            rx, ry, rz: Euler angles in radians (rotations around X, Y, Z axes)
        """
        # From R = Rz @ Ry @ Rx, we have:
        # R[2,0] = -sin(ry)
        sin_ry = -R[2, 0]

        # Check for gimbal lock
        if abs(sin_ry) >= 1.0 - 1e-6:
            # Gimbal lock case
            ry = np.arcsin(np.clip(sin_ry, -1.0, 1.0))

            if sin_ry > 0:  # ry ≈ π/2
                rz = 0.0
                rx = np.arctan2(R[0, 1], R[1, 1])
            else:  # ry ≈ -π/2
                rz = 0.0
                rx = np.arctan2(-R[0, 1], R[1, 1])
        else:
            # Normal case - no gimbal lock
            ry = np.arcsin(np.clip(sin_ry, -1.0, 1.0))

            # Calculate rx from R[2,1] and R[2,2]
            rx = np.arctan2(R[2, 1], R[2, 2])

            # Calculate rz from R[1,0] and R[0,0]
            rz = np.arctan2(R[1, 0], R[0, 0])

        return rx, ry, rz

    def apply_velocity(self, v, dt=0.1):
        """
        Update camera pose based on velocity command.

        This implements the pose update for IBVS navigation:
        - Linear velocity is applied in camera frame, then transformed to world frame
        - Angular velocity is applied as incremental rotation in camera frame

        Args:
            v: Velocity command [vx, vy, vz, wx, wy, wz]
               vx, vy, vz: linear velocity in camera frame (m/s)
               wx, wy, wz: angular velocity in camera frame (rad/s)
            dt: Time step (default 0.1)

        Returns:
            new_pose: Updated pose [x, y, z, rx, ry, rz]
        """
        x, y, z, rx, ry, rz = self.pose
        vx, vy, vz, wx, wy, wz = v

        # Get current rotation matrix (angles are in radians)
        R_current = self.get_rotation_matrix()

        # Linear velocity: camera frame -> world frame
        v_camera = np.array([vx, vy, vz]) * dt
        v_world = R_current @ v_camera

        # Update position in world frame
        new_x = x + v_world[0]
        new_y = y + v_world[1]
        new_z = z + v_world[2]

        # Angular velocity update (small angle increments in radians)
        delta_rx, delta_ry, delta_rz = np.array([wx, wy, wz]) * dt

        # Create incremental rotation matrix for the angular velocity
        cx, sx = np.cos(delta_rx), np.sin(delta_rx)
        cy, sy = np.cos(delta_ry), np.sin(delta_ry)
        cz, sz = np.cos(delta_rz), np.sin(delta_rz)

        R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

        R_delta = R_z @ R_y @ R_x

        # Update rotation: R_new = R_current * R_delta (rotation in camera frame)
        R_new = R_current @ R_delta

        # Convert back to Euler angles
        new_rx, new_ry, new_rz = self.rotation_matrix_to_euler(R_new)

        # Create new pose
        new_pose = np.array([new_x, new_y, new_z, new_rx, new_ry, new_rz])

        # Update internal pose
        self.pose = new_pose

        return new_pose


# TODO: static kpts
