import cv2
import numpy as np
import itertools

from src.camera import Camera


def is_collinear_triplet(p1, p2, p3, eps=1e-3):
    area = 0.5 * abs(
        p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
    )
    return area < eps


def is_collinear_set(pts, eps=1e-3):
    if len(pts) < 3:
        return False
    for i, j, k in itertools.combinations(range(len(pts)), 3):
        if is_collinear_triplet(pts[i], pts[j], pts[k], eps):
            return True
    return False


class Matcher:
    def __init__(self, ref, tgt, camera=None):
        self.ref = ref
        self.tgt = tgt
        self.sift = cv2.SIFT_create()
        self.camera = camera

    def _empty_rgb_canvas(self):
        def ensure_rgb(img):
            if img is None:
                return np.zeros((1, 1, 3), dtype=np.uint8)
            if img.ndim == 2:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if img.shape[-1] == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img

        left_rgb = ensure_rgb(self.ref)
        right_rgb = ensure_rgb(self.tgt)

        height = max(left_rgb.shape[0], right_rgb.shape[0])
        width = left_rgb.shape[1] + right_rgb.shape[1]
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[: left_rgb.shape[0], : left_rgb.shape[1]] = left_rgb
        canvas[: right_rgb.shape[0], left_rgb.shape[1] :] = right_rgb
        return canvas

    def _to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    def match(self, ratio_threshold=0.75, max_draw_matches=None):
        if self.ref is None or self.tgt is None:
            return {
                "good_matches": [],
                "ref_keypoints": [],
                "tgt_keypoints": [],
                "pts_ref": np.zeros((0, 2), dtype=np.float32),
                "pts_tgt": np.zeros((0, 2), dtype=np.float32),
                "matched_image": self._empty_rgb_canvas(),
                "raw_match_count": 0,
                "ratio_threshold": ratio_threshold,
            }

        gray1 = self._to_gray(self.ref)
        gray2 = self._to_gray(self.tgt)

        ref_kpts, ref_desc = self.sift.detectAndCompute(gray1, None)
        tgt_kpts, tgt_desc = self.sift.detectAndCompute(gray2, None)

        ref_kpts = ref_kpts or []
        tgt_kpts = tgt_kpts or []

        if ref_desc is None or tgt_desc is None:
            return {
                "good_matches": [],
                "ref_keypoints": ref_kpts,
                "tgt_keypoints": tgt_kpts,
                "pts_ref": np.zeros((0, 2), dtype=np.float32),
                "pts_tgt": np.zeros((0, 2), dtype=np.float32),
                "matched_image": self._empty_rgb_canvas(),
                "raw_match_count": 0,
                "ratio_threshold": ratio_threshold,
            }

        index_params = dict(algorithm=1, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(ref_desc, tgt_desc, k=2)

        good_matches = []
        for m, n in matches:
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)

        good_matches = sorted(good_matches, key=lambda m: m.distance)

        pts1 = np.float32([ref_kpts[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([tgt_kpts[m.trainIdx].pt for m in good_matches])

        matches_to_draw = (
            good_matches
            if max_draw_matches is None
            else good_matches[:max_draw_matches]
        )

        img_matches = cv2.drawMatches(
            self.ref,
            ref_kpts,
            self.tgt,
            tgt_kpts,
            matches_to_draw,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
        )

        return {
            "good_matches": good_matches,
            "ref_keypoints": ref_kpts,
            "tgt_keypoints": tgt_kpts,
            "pts_ref": pts1,
            "pts_tgt": pts2,
            "matched_image": cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB),
            "raw_match_count": len(matches),
            "ratio_threshold": ratio_threshold,
        }

    def query_corr_depth(self, corr, ref_depth_map, tgt_depth_map):
        pts1, pts2 = corr
        N = len(pts1)

        def sample_depth(pts, depth_map):
            samples = np.zeros((N, 3), dtype=float)
            if depth_map is None:
                return samples
            h, w = depth_map.shape[:2]
            for i, (u, v) in enumerate(pts):
                ui = int(np.clip(round(u), 0, w - 1))
                vi = int(np.clip(round(v), 0, h - 1))
                samples[i] = [u, v, float(depth_map[vi, ui])]
            return samples

        ref_points = sample_depth(pts1, ref_depth_map)
        tgt_points = sample_depth(pts2, tgt_depth_map)
        return ref_points, tgt_points

    def filter_matches_with_ransac(
        self,
        pts_ref,
        pts_tgt,
        distance_threshold=5.0,
        collinearity_eps=1e-3,
        max_iterations=1000,
    ):
        if len(pts_ref) < 4:
            return {
                "pts_ref": pts_ref,
                "pts_tgt": pts_tgt,
                "inlier_mask": np.ones(len(pts_ref), dtype=bool),
                "num_inliers": len(pts_ref),
                "num_outliers": 0,
                "H": None,
            }

        best_H = None
        best_inliers_count = 0
        best_inlier_mask = np.zeros(len(pts_ref), dtype=bool)
        N = len(pts_ref)
        n_samples = 4

        for _ in range(max_iterations):
            sample_idx = np.random.choice(N, n_samples, replace=False)
            sample_ref = pts_ref[sample_idx]
            sample_tgt = pts_tgt[sample_idx]

            if is_collinear_set(sample_ref, eps=collinearity_eps):
                continue
            if is_collinear_set(sample_tgt, eps=collinearity_eps):
                continue

            try:
                H, _ = cv2.findHomography(sample_ref, sample_tgt, 0)
                if H is None:
                    continue
            except:
                continue

            pts_ref_h = np.hstack([pts_ref, np.ones((N, 1))])
            proj = (H @ pts_ref_h.T).T
            proj = proj[:, :2] / proj[:, 2:3]

            d = np.linalg.norm(proj - pts_tgt, axis=1)
            inliers = d < distance_threshold
            count = np.sum(inliers)

            if count > best_inliers_count:
                best_inliers_count = count
                best_H = H.copy()
                best_inlier_mask = inliers.copy()

        filtered_ref = pts_ref[best_inlier_mask]
        filtered_tgt = pts_tgt[best_inlier_mask]

        return {
            "pts_ref": filtered_ref,
            "pts_tgt": filtered_tgt,
            "inlier_mask": best_inlier_mask,
            "num_inliers": len(filtered_ref),
            "num_outliers": len(pts_ref) - len(filtered_ref),
            "H": best_H,
        }

    def select_non_collinear(self, pts_ref, pts_tgt, max_pts=4):
        idxs = np.arange(len(pts_ref))
        for subset in itertools.combinations(idxs, max_pts):
            ref_subset = pts_ref[list(subset)]
            tgt_subset = pts_tgt[list(subset)]
            if not is_collinear_set(ref_subset) and not is_collinear_set(tgt_subset):
                return ref_subset, tgt_subset
        return pts_ref[:max_pts], pts_tgt[:max_pts]
