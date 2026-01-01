import numpy as np
import cv2
import matplotlib.pyplot as plt


def correspond(img1, img2):
    img1_cv = np.asarray(img1)
    img2_cv = np.asarray(img2)
    if img1_cv.dtype != np.uint8:
        img1_cv = img1_cv.astype(np.uint8)
    if img2_cv.dtype != np.uint8:
        img2_cv = img2_cv.astype(np.uint8)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_cv, None)
    kp2, des2 = sift.detectAndCompute(img2_cv, None)

    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return kp1 or [], kp2 or [], []

    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)
    return kp1, kp2, [m for m, n in matches if m.distance < 0.7 * n.distance]


def filter_matches_ransac(kp1, kp2, matches):
    if len(matches) < 4:
        return []
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return [m for m, val in zip(matches, mask.ravel()) if val == 1]


def select_best_4(kp1, kp2, matches):
    """Select the 4 best matches by descriptor distance."""
    if len(matches) < 4:
        return [], [], []

    sorted_matches = sorted(matches, key=lambda x: x.distance)[:4]

    new_kp1 = [kp1[m.queryIdx] for m in sorted_matches]
    new_kp2 = [kp2[m.trainIdx] for m in sorted_matches]
    new_matches = [cv2.DMatch(i, i, m.distance) for i, m in enumerate(sorted_matches)]

    return new_kp1, new_kp2, new_matches


def compute_euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def get_rotation_matrix_from_euler(rx, ry, rz):
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    return R_z @ R_y @ R_x


def get_relative_transform(pose1, pose2):
    x1, y1, z1, rx1, ry1, rz1 = pose1
    x2, y2, z2, rx2, ry2, rz2 = pose2

    R1 = get_rotation_matrix_from_euler(rx1, ry1, rz1)
    t1 = np.array([x1, y1, z1])

    R2 = get_rotation_matrix_from_euler(rx2, ry2, rz2)
    t2 = np.array([x2, y2, z2])

    R_rel = R2.T @ R1
    t_rel = R2.T @ (t1 - t2)

    return R_rel, t_rel


def filter_by_reprojection_error(
    cam, pose1, pose2, keypoints1, keypoints2, matches, depth1, threshold=200.0
):
    """
    Filter matches by 3D reprojection error.

    Args:
        cam: Camera object (for intrinsics and projection functions)
        pose1: Pose of camera 1 (query) [x, y, z, rx, ry, rz]
        pose2: Pose of camera 2 (target) [x, y, z, rx, ry, rz]
        keypoints1: Keypoints from image 1
        keypoints2: Keypoints from image 2
        matches: SIFT matches
        depth1: Depth map from camera 1
        threshold: Reprojection error threshold in pixels

    Returns:
        filtered_kp1: List of keypoints from image 1 that passed filtering
        filtered_kp2: List of keypoints from image 2 that passed filtering
        filtered_matches: List of geometrically consistent matches
    """
    depth_arr = np.asarray(depth1)
    filtered_matches = []
    filtered_kp1 = []
    filtered_kp2 = []

    R_rel, t_rel = get_relative_transform(pose1, pose2)

    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        u1, v1 = keypoints1[idx1].pt
        u2, v2 = keypoints2[idx2].pt

        v1_int, u1_int = int(round(v1)), int(round(u1))
        if not (0 <= v1_int < depth_arr.shape[0] and 0 <= u1_int < depth_arr.shape[1]):
            continue

        z1 = depth_arr[v1_int, u1_int]

        if not np.isfinite(z1) or z1 <= 0:
            continue

        point_cam1 = cam.backproject(np.array([u1, v1]), z1)
        point_cam2 = R_rel @ point_cam1 + t_rel

        if point_cam2[2] <= 0:
            continue

        projected_pt = cam.project(point_cam2)

        if projected_pt is None:
            continue

        if not (0 <= projected_pt[0] < cam.width and 0 <= projected_pt[1] < cam.height):
            continue

        reprojection_error = compute_euclidean_distance(projected_pt, [u2, v2])

        if reprojection_error < threshold:
            new_idx = len(filtered_kp1)
            new_match = cv2.DMatch(new_idx, new_idx, match.distance)
            filtered_matches.append(new_match)
            filtered_kp1.append(keypoints1[idx1])
            filtered_kp2.append(keypoints2[idx2])

    return filtered_kp1, filtered_kp2, filtered_matches


def plot_matches(img1, kp1, img2, kp2, matches):
    img1_cv = np.asarray(img1)
    img2_cv = np.asarray(img2)
    if img1_cv.dtype != np.uint8:
        img1_cv = img1_cv.astype(np.uint8)
    if img2_cv.dtype != np.uint8:
        img2_cv = img2_cv.astype(np.uint8)
    kp1_list = list(kp1) if not isinstance(kp1, list) else kp1
    kp2_list = list(kp2) if not isinstance(kp2, list) else kp2
    matches_list = list(matches) if not isinstance(matches, list) else matches

    res = cv2.drawMatches(
        img1_cv,
        kp1_list,
        img2_cv,
        kp2_list,
        matches_list,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    plt.figure(figsize=(15, 8))
    plt.imshow(res)
    plt.title(f"Feature Matches ({len(matches_list)} matches)")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def full_pipeline(
    cam,
    ref,
    tgt,
    ref_depth,
    tgt_depth,
    pose_ref,
    pose_tgt,
    reproj_threshold=200.0,
):
    kpts1, kpts2, correspondences = correspond(ref, tgt)

    fkpts1, fkpts2, filtered_matches = filter_by_reprojection_error(
        cam,
        pose_ref,
        pose_tgt,
        kpts1,
        kpts2,
        correspondences,
        ref_depth,
        reproj_threshold,
    )

    ransac_filtered = filter_matches_ransac(fkpts1, fkpts2, filtered_matches)

    best_kp1, best_kp2, best_matches = select_best_4(fkpts1, fkpts2, ransac_filtered)

    return best_kp1, best_kp2, best_matches
