import open3d as o3d
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse
from rich import print


from src.scene import SceneLoader
from src.camera import Camera
from src.features import full_pipeline, get_relative_transform
from src.control import compute_current_L, velocity, compute_desired_L, compute_normalized_L


def run_ibvs_simulation(
    glb_filename,
    desired_pose,
    initial_pose,
    width=1280,
    height=720,
    max_iterations=500,
    error_threshold=1.0,
    dt=0.1,
    debug=False,
):
    loader = SceneLoader(glb_filename)
    model = loader.load()

    render = o3d.visualization.rendering.OffscreenRenderer(width, height)
    render.scene.set_background([0.8, 0.8, 0.8, 1.0])
    render.scene.set_lighting(
        render.scene.LightingProfile.MED_SHADOWS, [-0.577, -0.577, -0.577]
    )
    render.scene.add_model("my_model", model)

    cam = Camera(pose=desired_pose, width=width, height=height)
    desired_view = cam.get_view(render)
    desired_depth = cam.capture_depth(render)

    cam.update_pose(initial_pose)
    initial_view = cam.get_view(render)
    initial_depth = cam.capture_depth(render)

    kpts_desired, kpts_initial, correspondences = full_pipeline(
        cam,
        desired_view,
        initial_view,
        desired_depth,
        initial_depth,
        desired_pose,
        initial_pose,
    )

    desired_points_3d, desired_pixels = cam.backproject_points(
        kpts_desired, desired_depth
    )
    current_points_3d, current_pixels = cam.backproject_points(
        kpts_initial, initial_depth
    )

    L = compute_current_L(cam, current_points_3d)
    v, err_norm = velocity(current_pixels, desired_pixels, L)

    trajectory = [initial_pose.copy()]
    errors = [err_norm]
    velocities = [v.copy()]
    views = [initial_view.copy()]

    plt.ion()
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    ax_view = fig.add_subplot(gs[0, 0])
    ax_error = fig.add_subplot(gs[1, 0])
    ax_vel = fig.add_subplot(gs[0, 1])
    ax_3d = fig.add_subplot(gs[1, 1], projection="3d")

    ax_view.set_title("Current Camera View")
    ax_view.axis("off")

    ax_error.set_title("Error Norm Evolution")
    ax_error.set_xlabel("Iteration")
    ax_error.set_ylabel("Error (pixels)")
    ax_error.grid(True)

    ax_vel.set_title("Velocity Commands Evolution")
    ax_vel.set_xlabel("Iteration")
    ax_vel.set_ylabel("Velocity")
    ax_vel.grid(True)

    ax_3d.set_title("3D Camera Trajectory")
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")

    iteration = 0
    converged = False

    while err_norm > error_threshold and iteration < max_iterations:
        iteration += 1

        new_pose = cam.apply_velocity(v, dt=dt)
        cam.update_pose(new_pose)
        trajectory.append(new_pose.copy())

        current_view = cam.get_view(render)
        current_depth = cam.capture_depth(render)
        views.append(current_view.copy())

        current_pixels_projected = []
        current_points_3d_updated = []

        R_rel, t_rel = get_relative_transform(desired_pose, new_pose)

        for i, point_3d_desired in enumerate(desired_points_3d):
            point_3d_current = R_rel @ point_3d_desired + t_rel
            pixel_projected = cam.project(point_3d_current)

            if pixel_projected is not None:
                u, v = pixel_projected
                if 0 <= u < cam.width and 0 <= v < cam.height:
                    current_pixels_projected.append(pixel_projected)
                    current_points_3d_updated.append(point_3d_current)
                else:
                    current_pixels_projected.append(pixel_projected)
                    current_points_3d_updated.append(point_3d_current)
            else:
                current_pixels_projected.append(desired_pixels[i])
                current_points_3d_updated.append(point_3d_desired)

        current_pixels = np.array(current_pixels_projected)
        current_points_3d = np.array(current_points_3d_updated)

        current_depth_arr = np.asarray(current_depth)
        current_depths = []
        for pixel in current_pixels:
            u, v = int(round(pixel[0])), int(round(pixel[1]))
            if 0 <= u < cam.width and 0 <= v < cam.height:
                depth = current_depth_arr[v, u]
                if np.isfinite(depth) and depth > 0:
                    current_depths.append(depth)
                else:
                    current_depths.append(current_points_3d[len(current_depths)][2])
            else:
                current_depths.append(current_points_3d[len(current_depths)][2])

        for i, depth in enumerate(current_depths):
            current_points_3d[i] = cam.backproject(current_pixels[i], depth)

        L = compute_normalized_L(cam, current_points_3d, desired_points_3d)
        v, err_norm = velocity(current_pixels, desired_pixels, L)
        errors.append(err_norm)
        velocities.append(v.copy())

        ax_view.clear()
        ax_view.imshow(cv2.cvtColor(current_view, cv2.COLOR_BGR2RGB))
        ax_view.set_title(f"Current Camera View (Iteration {iteration})")
        ax_view.axis("off")

        ax_error.clear()
        ax_error.plot(errors, "b-", linewidth=2)
        ax_error.axhline(
            error_threshold, color="r", linestyle="--", linewidth=2, label="Threshold"
        )
        ax_error.set_xlabel("Iteration")
        ax_error.set_ylabel("Error (pixels)")
        ax_error.set_title("Error Norm Evolution")
        ax_error.legend()
        ax_error.grid(True)
        ax_error.set_yscale("log")

        vel_array = np.array(velocities)
        ax_vel.clear()
        ax_vel.plot(vel_array[:, 0], label="vx", linewidth=1.5)
        ax_vel.plot(vel_array[:, 1], label="vy", linewidth=1.5)
        ax_vel.plot(vel_array[:, 2], label="vz", linewidth=1.5)
        ax_vel.plot(vel_array[:, 3], label="wx", linewidth=1.5)
        ax_vel.plot(vel_array[:, 4], label="wy", linewidth=1.5)
        ax_vel.plot(vel_array[:, 5], label="wz", linewidth=1.5)
        ax_vel.set_xlabel("Iteration")
        ax_vel.set_ylabel("Velocity")
        ax_vel.set_title("Velocity Commands Evolution")
        ax_vel.legend(ncol=2, fontsize=8)
        ax_vel.grid(True)

        ax_3d.clear()
        traj_array = np.array(trajectory)
        ax_3d.plot(
            traj_array[:, 0],
            traj_array[:, 1],
            traj_array[:, 2],
            "b-o",
            markersize=2,
            linewidth=1.5,
        )
        ax_3d.scatter(
            desired_pose[0],
            desired_pose[1],
            desired_pose[2],
            c="r",
            marker="*",
            s=300,
            label="Desired",
        )
        ax_3d.scatter(
            initial_pose[0],
            initial_pose[1],
            initial_pose[2],
            c="g",
            marker="*",
            s=300,
            label="Initial",
        )
        ax_3d.scatter(
            new_pose[0],
            new_pose[1],
            new_pose[2],
            c="b",
            marker="o",
            s=150,
            label="Current",
        )
        ax_3d.set_xlabel("X")
        ax_3d.set_ylabel("Y")
        ax_3d.set_zlabel("Z")
        ax_3d.set_title("3D Camera Trajectory")
        ax_3d.legend()
        ax_3d.grid(True)

        plt.pause(0.01)

        if debug:
            print(
                f"[green]Iteration {iteration}[/green]: "
                f"[red]Error = {err_norm:.2f}[/red], "
                f"[blue]|v| = {np.linalg.norm(v):.4f}[/blue]"
            )

    if err_norm <= error_threshold:
        converged = True
        print(f"\n✓ Converged in {iteration} iterations!")
        print(f"Final error: {err_norm:.2f} pixels")
    else:
        print(f"\n✗ Did not converge after {max_iterations} iterations")
        print(f"Final error: {err_norm:.2f} pixels")

    print(f"Final pose: {cam.pose}")
    print(f"Desired pose: {desired_pose}")

    plt.ioff()
    plt.show()

    return {
        "converged": converged,
        "iterations": iteration,
        "final_error": err_norm,
        "trajectory": np.array(trajectory),
        "errors": np.array(errors),
        "velocities": np.array(velocities),
        "final_pose": cam.pose,
        "views": views,
    }



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    glb_filename = "data/modular_environment.glb"
    desired_pose = [85.0, 80.0, -20.0, -0.4, 0.6, 0.0]
    initial_pose = [85.0, 80.0, -25.0, -0.4, 0.6, 0.0]

    results = run_ibvs_simulation(
        glb_filename,
        desired_pose,
        initial_pose,
        width=1280,
        height=720,
        max_iterations=500,
        error_threshold=1.0,
        dt=0.1,
        debug=args.debug,
    )
