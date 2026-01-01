import numpy as np
import cv2
import itertools
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
    Filter matches by 3D reprojection error and return filtered keypoints.

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
        filtered_matches: List of geometrically consistent matches
        filtered_kp1: List of keypoints from image 1 that passed filtering
        filtered_kp2: List of keypoints from image 2 that passed filtering
    """
    depth_arr = np.asarray(depth1)
    filtered_matches = []
    filtered_kp1 = []
    filtered_kp2 = []

    # Get relative transformation from cam1 to cam2
    R_rel, t_rel = get_relative_transform(pose1, pose2)

    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx

        # Get pixel coordinates
        u1, v1 = keypoints1[idx1].pt
        u2, v2 = keypoints2[idx2].pt

        # Get depth (check bounds)
        v1_int, u1_int = int(round(v1)), int(round(u1))
        if not (0 <= v1_int < depth_arr.shape[0] and 0 <= u1_int < depth_arr.shape[1]):
            continue

        z1 = depth_arr[v1_int, u1_int]

        # Skip invalid depths
        if not np.isfinite(z1) or z1 <= 0:
            continue

        # Backproject to 3D in camera 1 frame
        point_cam1 = cam.backproject(np.array([u1, v1]), z1)

        # Transform to camera 2 frame
        point_cam2 = R_rel @ point_cam1 + t_rel

        # Skip if behind camera
        if point_cam2[2] <= 0:
            continue

        # Project to image 2
        projected_pt = cam.project(point_cam2)

        if projected_pt is None:
            continue

        # Check bounds
        if not (0 <= projected_pt[0] < cam.width and 0 <= projected_pt[1] < cam.height):
            continue

        # Calculate reprojection error
        reprojection_error = compute_euclidean_distance(projected_pt, [u2, v2])

        # Filter by threshold
        if reprojection_error < threshold:
            new_idx = len(filtered_kp1)
            new_match = cv2.DMatch(new_idx, new_idx, match.distance)
            filtered_matches.append(new_match)
            filtered_kp1.append(keypoints1[idx1])
            filtered_kp2.append(keypoints2[idx2])

    return filtered_kp1, filtered_kp2, filtered_matches


def _reindex_matches(matches, keypoints1, keypoints2):
    """Re-index matches to 0,1,2,... and return (matches, kp1, kp2)."""
    new_matches = []
    new_kp1 = []
    new_kp2 = []
    for i, m in enumerate(matches):
        new_matches.append(cv2.DMatch(i, i, m.distance))
        new_kp1.append(keypoints1[m.queryIdx])
        new_kp2.append(keypoints2[m.trainIdx])
    return new_matches, new_kp1, new_kp2


def select_best_4(
    keypoints1, keypoints2, matches, min_area=100.0
):
    if len(matches) < 4:
        print(f"Error: Need at least 4 matches, got {len(matches)}")
        return [], [], []

    # Sort matches by descriptor distance (lower is better)
    sorted_matches = sorted(matches, key=lambda x: x.distance)

    # Try to find the best 4 non-collinear matches
    # Start with the top 4, if collinear, try next combinations

    if len(matches) == 4:
        # Only 4 matches, verify non-collinearity
        kp1_pts = np.array(
            [keypoints1[m.queryIdx].pt for m in sorted_matches], dtype=np.float32
        )
        kp2_pts = np.array(
            [keypoints2[m.trainIdx].pt for m in sorted_matches], dtype=np.float32
        )

        if is_non_collinear(kp1_pts, min_area) and is_non_collinear(kp2_pts, min_area):
            return _reindex_matches(sorted_matches, keypoints1, keypoints2)
        else:
            print("Warning: Only 4 matches available but they are collinear!")
            return [], [], []

    # Try combinations starting from the best matches
    # Strategy: Take top N candidates, find best 4 non-collinear subset
    n_candidates = min(len(matches), 10)  # Look at top 10 matches
    candidates = sorted_matches[:n_candidates]

    # First, try the simplest case: top 4 matches
    top_4 = candidates[:4]
    kp1_pts = np.array([keypoints1[m.queryIdx].pt for m in top_4], dtype=np.float32)
    kp2_pts = np.array([keypoints2[m.trainIdx].pt for m in top_4], dtype=np.float32)

    if is_non_collinear(kp1_pts, min_area) and is_non_collinear(kp2_pts, min_area):
        return _reindex_matches(top_4, keypoints1, keypoints2)

    # If top 4 are collinear, search for best non-collinear combination
    # among top candidates
    best_combo = None
    best_avg_distance = float("inf")

    for combo in itertools.combinations(range(len(candidates)), 4):
        indices = list(combo)
        combo_matches = [candidates[i] for i in indices]

        kp1_pts = np.array(
            [keypoints1[m.queryIdx].pt for m in combo_matches], dtype=np.float32
        )
        kp2_pts = np.array(
            [keypoints2[m.trainIdx].pt for m in combo_matches], dtype=np.float32
        )

        # Check non-collinearity
        if is_non_collinear(kp1_pts, min_area) and is_non_collinear(kp2_pts, min_area):
            # Compute average descriptor distance
            avg_distance = np.mean([m.distance for m in combo_matches])

            if avg_distance < best_avg_distance:
                best_avg_distance = avg_distance
                best_combo = combo_matches

    if best_combo is None:
        print("Warning: Could not find 4 non-collinear matches among top candidates!")
        return _select_greedy_from_best(
            keypoints1, keypoints2, sorted_matches, min_area
        )

    return _reindex_matches(best_combo, keypoints1, keypoints2)


def is_non_collinear(points, min_area=100.0):
    """
    Check if 4 points are non-collinear by computing convex hull area.

    Args:
        points: Array of points [[x1,y1], [x2,y2], ...]
        min_area: Minimum acceptable area

    Returns:
        True if points are sufficiently non-collinear
    """
    if len(points) < 3:
        return False

    if len(points) == 3:
        # For 3 points, compute triangle area
        hull = cv2.convexHull(points, returnPoints=True)
        area = cv2.contourArea(hull)
        return area > min_area

    # For 4+ points, compute convex hull area
    hull = cv2.convexHull(points, returnPoints=True)
    area = cv2.contourArea(hull)

    # Also check that points aren't too close together
    min_dist = float("inf")
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = np.linalg.norm(points[i] - points[j])
            min_dist = min(min_dist, dist)

    return area > min_area and min_dist > 10.0  # At least 10 pixels apart

def _select_greedy_from_best(keypoints1, keypoints2, sorted_matches, min_area):
    """
    Greedy fallback: iteratively pick matches that maximize spatial distribution.
    """
    selected = []

    for match in sorted_matches:
        if len(selected) >= 4:
            break

        # Try adding this match
        test_matches = selected + [match]
        kp1_pts = np.array(
            [keypoints1[m.queryIdx].pt for m in test_matches], dtype=np.float32
        )

        if len(kp1_pts) < 3:
            # Not enough points to check collinearity yet
            selected.append(match)
        else:
            kp2_pts = np.array(
                [keypoints2[m.trainIdx].pt for m in test_matches], dtype=np.float32
            )

            # Check if adding this point improves spatial distribution
            if len(kp1_pts) == 3:
                # For 3 points, just check they form a triangle
                hull = cv2.convexHull(kp1_pts, returnPoints=True)
                area = cv2.contourArea(hull)
                if area > min_area / 2:  # Less strict for 3 points
                    selected.append(match)
            else:
                # For 4 points, check full non-collinearity
                if is_non_collinear(kp1_pts, min_area) and is_non_collinear(
                    kp2_pts, min_area
                ):
                    selected.append(match)

    if len(selected) < 4:
        return [], [], []

    return _reindex_matches(selected[:4], keypoints1, keypoints2)
    

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

    # Draw matches with flags parameter
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
    min_area=100.0,
):
    kpts1, kpts2, correspondences = correspond(ref, tgt)
    fkpts1, fkpts2, filtered1 = filter_by_reprojection_error(
        cam,
        pose_ref,
        pose_tgt,
        kpts1,
        kpts2,
        correspondences,
        ref_depth,
        reproj_threshold,
    )
    filtered = filter_matches_ransac(fkpts1, fkpts2, filtered1)
    best_4, bkpts1, bkpts2 = select_best_4(fkpts1, fkpts2, filtered, min_area)
    # Return in order: kpts1, kpts2, matches
    return bkpts1, bkpts2, best_4
