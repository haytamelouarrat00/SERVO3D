import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

from src.utils import plot_images
from src.scene import SceneLoader
from src.camera import Camera
from src.features import correspond, filter_by_reprojection_error, filter_matches_ransac, select_best_4, plot_matches
#Debug flag
parser = argparse.ArgumentParser()
parser.add_argument("--debug", "-d", action="store_true")
args = parser.parse_args()
DEBUG = args.debug

# Setup
glb_filename = "data/modular_environment.glb"
width, height = 1280, 720

loader = SceneLoader(glb_filename)
model = loader.load()

#Render Init
render = o3d.visualization.rendering.OffscreenRenderer(width, height)
render.scene.set_background([0.8, 0.8, 0.8, 1.0])
render.scene.set_lighting(render.scene.LightingProfile.MED_SHADOWS, [-0.577, -0.577, -0.577])
render.scene.add_model("my_model", model)

#Camera + desired view
desired_pose = [80.0, 80.0, -20.0, -0.6, 0.78, 0.0]
cam = Camera(pose=desired_pose,
             width=width,
             height=height)
desired_view = cam.get_view(render)
desired_depth = cam.capture_depth(render)
#initial view
initial_pose = [80.0, 80.0, -50.0, -0.6, 0.78, 0.0]
cam.update_pose(initial_pose)
initial_view = cam.get_view(render)
initial_depth = cam.capture_depth(render)

kpts1, kpts2, correspondences = correspond(desired_view, initial_view)
fkpts1, fkpts2, filtered1 = filter_by_reprojection_error(
    cam, desired_pose, initial_pose, kpts1, kpts2, correspondences, desired_depth
)
filtered = filter_matches_ransac(fkpts1, fkpts2, filtered1)
best_4, bkpts1, bkpts2 = select_best_4(fkpts1, fkpts2, filtered)

if DEBUG:
    plot_images(desired_view, initial_view, "Desired View", "Initial View")
    print(f"Found {len(correspondences)} SIFT correspondences between desired and initial views.")
    plot_images(cam.visualize_depth(desired_depth), cam.visualize_depth(initial_depth), "Desired Depth", "Initial Depth")
    print(
        f"Filtered {len(filtered1)} from {len(correspondences)}, ratio = {len(filtered1) / len(correspondences) * 100}%"
    )
    print(
        f"Filtered {len(filtered)} from {len(filtered1)}, ratio = {len(filtered) / len(filtered1) * 100}% "
    )
    plot_matches(desired_view, bkpts1, initial_view, bkpts2, best_4)
    