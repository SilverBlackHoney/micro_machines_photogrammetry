
import cv2
import numpy as np
from scipy.optimize import least_squares
import open3d as o3d
from typing import List, Tuple, Optional
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt


@dataclass
class CameraParameters:
    """Camera intrinsic parameters"""
    focal_length: float
    principal_point: Tuple[float, float]
    distortion: np.ndarray
    image_size: Tuple[int, int]


@dataclass
class CameraPose:
    """Camera extrinsic parameters"""
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector
    image_id: int


class PhotogrammetryPipeline:
    """Photogrammetry pipeline for Structure from Motion (SfM)"""

    def __init__(self, images_folder: str):
        if not os.path.exists(images_folder):
            raise ValueError(f"Images folder does not exist: {images_folder}")

        self.images_folder = images_folder
        self.images = []
        self.keypoints = []
        self.descriptors = []
        self.matches = {}  # Initialize matches to prevent KeyErrors
        self.camera_params = None
        self.camera_poses = {}
        self.point_cloud = []
        self.point_colors = []
        self.track_graph = {}
        self.depth_maps = {}

    def _create_detector(self, feature_type='ORB', use_cuda=False):
        """Create feature detector with proper CUDA handling"""
        use_cuda_detector = False
        detector_name = feature_type

        if feature_type == 'SIFT':
            if hasattr(cv2, 'SIFT_create'):
                detector = cv2.SIFT_create(nfeatures=2000)
                detector_name = "SIFT"
            else:
                raise ValueError("SIFT is not available in your OpenCV installation. Please install opencv-contrib-python.")
        elif feature_type == 'ORB':
            if use_cuda and hasattr(cv2, 'cuda') and cv2.cuda.getCudaEnabledDeviceCount() > 0:
                try:
                    detector = cv2.cuda.ORB_create(nfeatures=2000)
                    use_cuda_detector = True
                    detector_name = "ORB (CUDA)"
                except Exception:
                    # Fallback to CPU if CUDA fails
                    detector = cv2.ORB_create(nfeatures=2000)
                    detector_name = "ORB (CPU fallback)"
            else:
                detector = cv2.ORB_create(nfeatures=2000)
                detector_name = "ORB"
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        return detector, use_cuda_detector, detector_name

    def _get_matcher_for_descriptor(self, feature_type):
        """Get appropriate matcher for descriptor type"""
        if feature_type.upper() in ['SIFT', 'SURF']:
            return cv2.BFMatcher(cv2.NORM_L2)
        elif feature_type.upper() in ['ORB', 'BRISK', 'AKAZE']:
            return cv2.BFMatcher(cv2.NORM_HAMMING)
        else:
            return cv2.BFMatcher(cv2.NORM_L2)  # Default fallback

    def load_images(self) -> List[np.ndarray]:
        """Load all images from folder"""
        image_files = [
            f for f in os.listdir(self.images_folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        image_files.sort()

        for img_file in image_files:
            img_path = os.path.join(self.images_folder, img_file)
            img = cv2.imread(img_path)
            if img is not None:
                self.images.append(img)

        print(f"Loaded {len(self.images)} images")
        return self.images

    def extract_features(self, feature_type='ORB', use_cuda=False):
        """Extract features from all images"""
        detector, use_cuda_detector, detector_name = self._create_detector(feature_type, use_cuda)

        for img in self.images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if use_cuda_detector:
                gpu_img = cv2.cuda_GpuMat()
                gpu_img.upload(gray)
                kps = detector.detect(gpu_img)
                kps, descs = detector.compute(gpu_img, kps)
                descs = descs.download() if descs is not None else None
                # Release GPU memory
                del gpu_img
            else:
                kps, descs = detector.detectAndCompute(gray, None)
            self.keypoints.append(kps)
            self.descriptors.append(descs)

        print(f"Extracted features from {len(self.images)} images using {detector_name}")

    def match_features(self, feature_type='ORB', ratio_threshold=0.7):
        """Match features between all image pairs"""
        matcher = self._get_matcher_for_descriptor(feature_type)

        for i in range(len(self.images)):
            for j in range(i + 1, len(self.images)):
                if self.descriptors[i] is not None and self.descriptors[j] is not None:
                    raw_matches = matcher.knnMatch(
                        self.descriptors[i], self.descriptors[j], k=2
                    )

                    # Apply Lowe's ratio test
                    good_matches = []
                    for match_pair in raw_matches:
                        if len(match_pair) == 2:
                            m, n = match_pair
                            if m.distance < ratio_threshold * n.distance:
                                good_matches.append(m)

                    if len(good_matches) > 20:  # Minimum matches threshold
                        self.matches[(i, j)] = good_matches

        print(f"Found {len(self.matches)} valid image pairs")

    def estimate_camera_intrinsics(self, focal_length_guess=None):
        """Estimate camera intrinsic parameters"""
        if focal_length_guess is None:
            # Rough estimation based on image size
            h, w = self.images[0].shape[:2]
            focal_length_guess = max(w, h) * 1.2

        # Use first image dimensions
        h, w = self.images[0].shape[:2]
        principal_point = (w / 2, h / 2)

        self.camera_params = CameraParameters(
            focal_length=focal_length_guess,
            principal_point=principal_point,
            distortion=np.zeros(5),
            image_size=(w, h)
        )

        print(f"Estimated focal length: {focal_length_guess:.2f}")

    def estimate_fundamental_matrix(self, img1_idx: int, img2_idx: int) -> np.ndarray:
        """Estimate fundamental matrix between two images"""
        matches = self.matches.get((img1_idx, img2_idx), [])
        if len(matches) < 8:
            return None

        # Get matching points
        pts1 = np.float32(
            [self.keypoints[img1_idx][m.queryIdx].pt for m in matches]
        ).reshape(-1, 1, 2)
        pts2 = np.float32(
            [self.keypoints[img2_idx][m.trainIdx].pt for m in matches]
        ).reshape(-1, 1, 2)

        # Estimate fundamental matrix with RANSAC
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99)

        # Filter matches based on inliers
        inlier_matches = [matches[i] for i in range(len(matches)) if mask.ravel()[i]]
        self.matches[(img1_idx, img2_idx)] = inlier_matches

        return F

    def triangulate_points(
        self,
        img1_idx: int,
        img2_idx: int,
        pose1: CameraPose,
        pose2: CameraPose
    ) -> np.ndarray:
        """Triangulate 3D points from two camera views"""
        matches = self.matches.get((img1_idx, img2_idx), [])
        if len(matches) < 10:
            return np.array([])

        # Get matching points
        pts1 = np.array(
            [self.keypoints[img1_idx][m.queryIdx].pt for m in matches]
        ).T
        pts2 = np.array(
            [self.keypoints[img2_idx][m.trainIdx].pt for m in matches]
        ).T

        # Ensure camera intrinsics are estimated
        if self.camera_params is None:
            self.estimate_camera_intrinsics()

        K = np.array([
            [self.camera_params.focal_length, 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length, self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])

        P1 = K @ np.hstack([pose1.rotation, pose1.translation.reshape(-1, 1)])
        P2 = K @ np.hstack([pose2.rotation, pose2.translation.reshape(-1, 1)])

        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T

    def initialize_two_view_reconstruction(self):
        """Initialize reconstruction with the best two-view pair"""
        best_pair = None
        best_score = 0

        # Find the pair with most inlier matches
        for pair, matches in self.matches.items():
            if len(matches) > best_score:
                best_score = len(matches)
                best_pair = pair

        if best_pair is None:
            raise ValueError("No valid image pairs found")

        img1_idx, img2_idx = best_pair
        print(
            f"Initializing with images {img1_idx} and {img2_idx} ({best_score} matches)"
        )

        # Estimate essential matrix
        matches = self.matches[best_pair]
        pts1 = np.array(
            [self.keypoints[img1_idx][m.queryIdx].pt for m in matches]
        )
        pts2 = np.array(
            [self.keypoints[img2_idx][m.trainIdx].pt for m in matches]
        )

        K = np.array([
            [self.camera_params.focal_length, 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length, self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])

        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)

        # Recover pose
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)

        # Set first camera at origin
        self.camera_poses[img1_idx] = CameraPose(
            rotation=np.eye(3),
            translation=np.zeros(3),
            image_id=img1_idx
        )

        # Set second camera pose relative to the first
        self.camera_poses[img2_idx] = CameraPose(
            rotation=R,
            translation=t.flatten(),
            image_id=img2_idx
        )

        # Triangulate initial points
        points_3d = self.triangulate_points(
            img1_idx, img2_idx,
            self.camera_poses[img1_idx],
            self.camera_poses[img2_idx]
        )

        # Initialize point cloud
        self.point_cloud = points_3d.tolist()

        # Create track graph (simplified)
        for i, match in enumerate(matches):
            if i < len(points_3d):
                track_id = len(self.track_graph)
                self.track_graph[track_id] = {
                    'point_3d': points_3d[i],
                    'observations': {
                        img1_idx: match.queryIdx,
                        img2_idx: match.trainIdx
                    }
                }

        print(f"Initialized with {len(self.point_cloud)} 3D points")

    def bundle_adjustment_residuals(
        self,
        params,
        observations,
        point_indices,
        camera_indices,
        img_ids
    ):
        """Compute residuals for bundle adjustment"""
        n_cameras = len(self.camera_poses)
        n_points = len(self.point_cloud)

        # Unpack parameters
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        K = np.array([
            [self.camera_params.focal_length, 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length, self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])

        residuals = []

        for obs, point_idx, cam_idx in zip(observations, point_indices, camera_indices):
            # Get camera pose from parameters
            rvec = camera_params[cam_idx][:3]
            tvec = camera_params[cam_idx][3:]

            # Project 3D point
            try:
                projected, _ = cv2.projectPoints(
                    points_3d[point_idx].reshape(1, 1, 3),
                    rvec, tvec, K, None
                )
                # Compute residual
                residual = obs - projected[0, 0]
                residuals.extend(residual)
            except cv2.error:
                # If projection fails, add large residual
                residuals.extend([100.0, 100.0])

        return np.array(residuals)

    def run_bundle_adjustment(self, max_iterations=50):
        """Run bundle adjustment optimization"""
        print("Running bundle adjustment...")

        # Create mapping from image_id to sequential index
        img_ids = sorted(self.camera_poses.keys())
        id_to_idx = {img_id: idx for idx, img_id in enumerate(img_ids)}

        # Prepare observations
        observations = []
        point_indices = []
        camera_indices = []

        for track_id, track in self.track_graph.items():
            for img_id, kp_idx in track['observations'].items():
                if img_id in self.camera_poses:
                    kp = self.keypoints[img_id][kp_idx]
                    observations.append([kp.pt[0], kp.pt[1]])
                    point_indices.append(track_id)
                    camera_indices.append(id_to_idx[img_id])

        observations = np.array(observations)

        # Initialize parameters
        n_cameras = len(self.camera_poses)
        n_points = len(self.point_cloud)

        camera_params = []
        for img_id in img_ids:
            pose = self.camera_poses[img_id]
            # Ensure pose.rotation is a valid 3x3 rotation matrix
            rot = pose.rotation
            if rot.shape != (3, 3):
                print(f"Warning: pose.rotation for image {img_id} is not 3x3, skipping.")
                continue
            # Orthonormalize the rotation matrix
            u, _, vh = np.linalg.svd(rot)
            rot_ortho = u @ vh
            if np.linalg.det(rot_ortho) < 0:
                rot_ortho *= -1
            try:
                rvec, _ = cv2.Rodrigues(rot_ortho)
            except cv2.error as e:
                print(f"cv2.Rodrigues failed for image {img_id}: {e}")
                continue
            camera_params.extend(rvec.flatten())
            camera_params.extend(pose.translation)

        points_params = np.array(self.point_cloud).flatten()
        initial_params = np.concatenate([camera_params, points_params])

        # Run optimization
        try:
            result = least_squares(
                self.bundle_adjustment_residuals,
                initial_params,
                args=(observations, point_indices, camera_indices, img_ids),
                max_nfev=max_iterations * len(initial_params)
            )

            if result.success:
                print(f"Bundle adjustment converged in {result.nfev} iterations")

                # Update camera poses and points from optimized parameters
                optimized_camera_params = result.x[:n_cameras * 6].reshape((n_cameras, 6))
                optimized_points = result.x[n_cameras * 6:].reshape((n_points, 3))

                # Update camera poses
                for i, img_id in enumerate(img_ids):
                    rvec = optimized_camera_params[i][:3]
                    tvec = optimized_camera_params[i][3:]
                    R, _ = cv2.Rodrigues(rvec)
                    self.camera_poses[img_id] = CameraPose(
                        rotation=R,
                        translation=tvec,
                        image_id=img_id
                    )

                # Update point cloud
                self.point_cloud = optimized_points.tolist()

                # Update track graph
                for track_id, new_point in enumerate(optimized_points):
                    if track_id in self.track_graph:
                        self.track_graph[track_id]['point_3d'] = new_point

            else:
                print("Bundle adjustment failed to converge")

        except Exception as e:
            print(f"Bundle adjustment error: {e}")

    def compute_depth_maps(self):
        """Compute dense depth maps for each camera using multi-view stereo"""
        print("Computing dense depth maps...")

        self.depth_maps = {}

        for img_id, pose in self.camera_poses.items():
            print(f"Processing depth map for image {img_id}")

            # Find neighboring cameras
            neighbors = self.find_neighboring_cameras(img_id, max_neighbors=5)

            if len(neighbors) < 2:
                continue

            # Compute depth map using plane sweeping
            depth_map = self.plane_sweep_stereo(img_id, neighbors)

            if depth_map is not None:
                self.depth_maps[img_id] = depth_map

        print(f"Computed {len(self.depth_maps)} depth maps")

    def find_neighboring_cameras(self, img_id: int, max_neighbors: int = 5) -> List[int]:
        """Find neighboring cameras based on baseline and viewing angle"""
        if img_id not in self.camera_poses:
            return []

        reference_pose = self.camera_poses[img_id]
        neighbors = []

        for other_id, other_pose in self.camera_poses.items():
            if other_id == img_id:
                continue

            # Calculate baseline (distance between cameras)
            baseline = np.linalg.norm(
                reference_pose.translation - other_pose.translation
            )

            # Calculate viewing angle difference
            ref_direction = reference_pose.rotation[:, 2]  # Z-axis (forward)
            other_direction = other_pose.rotation[:, 2]
            angle_diff = np.arccos(
                np.clip(np.dot(ref_direction, other_direction), -1, 1)
            )

            # Score based on baseline and angle (prefer moderate baseline, similar viewing)
            if 0.1 < baseline < 2.0 and angle_diff < np.pi / 4:  # Adjust thresholds for micro machines
                score = baseline * (1 - angle_diff / (np.pi / 4))
                neighbors.append((other_id, score))

        # Sort by score and return top neighbors
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [n[0] for n in neighbors[:max_neighbors]]

    def plane_sweep_stereo(self, ref_img_id: int, neighbor_ids: List[int], 
                          min_depth: float = 0.1, max_depth: float = 1.0, 
                          num_planes: int = 64) -> Optional[np.ndarray]:
        """Compute depth map using plane sweeping stereo"""
        if ref_img_id not in self.camera_poses:
            return None

        ref_img = self.images[ref_img_id]
        ref_pose = self.camera_poses[ref_img_id]
        h, w = ref_img.shape[:2]

        # Define depth planes with configurable range
        depths = np.linspace(min_depth, max_depth, num_planes)

        # Camera matrix
        K = np.array([
            [self.camera_params.focal_length, 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length, self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])

        # Cost volume for each depth plane
        cost_volume = np.zeros((h, w, num_planes))

        for d_idx, depth in enumerate(depths):
            costs = []

            for neighbor_id in neighbor_ids:
                if neighbor_id >= len(self.images):
                    continue

                neighbor_img = self.images[neighbor_id]
                neighbor_pose = self.camera_poses[neighbor_id]

                # Warp neighbor image to reference camera at current depth
                warped = self.warp_image_to_depth(
                    neighbor_img, neighbor_pose, ref_pose, depth, K
                )

                if warped is not None:
                    # Compute photometric cost (normalized cross correlation)
                    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
                    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

                    # Use window-based matching
                    window_size = 5
                    cost = self.compute_ncc_cost(
                        ref_gray, warped_gray, window_size
                    )
                    costs.append(cost)

            if costs:
                # Aggregate costs (mean)
                cost_volume[:, :, d_idx] = np.mean(costs, axis=0)

        # Winner-takes-all depth selection
        depth_indices = np.argmin(cost_volume, axis=2)
        depth_map = depths[depth_indices]

        # Post-process depth map
        return self.filter_depth_map(depth_map)

    def warp_image_to_depth(
        self,
        src_img: np.ndarray,
        src_pose: CameraPose,
        ref_pose: CameraPose,
        depth: float,
        K: np.ndarray
    ) -> Optional[np.ndarray]:
        """Warp source image to reference camera assuming constant depth"""
        h, w = src_img.shape[:2]

        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        ones = np.ones_like(x)

        # Convert to normalized coordinates
        coords = np.stack([x, y, ones], axis=-1).reshape(-1, 3)
        normalized_coords = (np.linalg.inv(K) @ coords.T).T

        # Project to 3D at given depth
        points_3d = normalized_coords * depth

        # Transform from reference to source camera
        # World to source camera coordinates
        R_rel = src_pose.rotation @ ref_pose.rotation.T
        t_rel = src_pose.translation - R_rel @ ref_pose.translation

        transformed_points = (R_rel @ points_3d.T).T + t_rel

        # Check for points behind camera
        if np.any(transformed_points[:, 2] <= 0):
            return None

        # Project to source image
        projected = (K @ transformed_points.T).T
        projected = projected[:, :2] / projected[:, 2:3]

        # Reshape back to image coordinates
        map_x = projected[:, 0].reshape(h, w).astype(np.float32)
        map_y = projected[:, 1].reshape(h, w).astype(np.float32)

        # Clip out-of-bounds and invalid coordinates
        map_x = np.clip(map_x, 0, w - 1)
        map_y = np.clip(map_y, 0, h - 1)
        map_x[np.isnan(map_x) | np.isinf(map_x)] = 0
        map_y[np.isnan(map_y) | np.isinf(map_y)] = 0

        # Warp image
        return cv2.remap(src_img, map_x, map_y, cv2.INTER_LINEAR)

    def compute_ncc_cost(self, img1: np.ndarray, img2: np.ndarray, window_size: int) -> np.ndarray:
        """Compute normalized cross-correlation cost"""
        # Convert to float
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0

        # Compute local means
        kernel = np.ones((window_size, window_size)) / (window_size ** 2)
        mean1 = cv2.filter2D(img1, -1, kernel)
        mean2 = cv2.filter2D(img2, -1, kernel)

        # Compute local standard deviations
        sqr1 = cv2.filter2D(img1 ** 2, -1, kernel)
        sqr2 = cv2.filter2D(img2 ** 2, -1, kernel)

        std1 = np.sqrt(np.maximum(sqr1 - mean1 ** 2, 1e-8))  # Avoid sqrt(0)
        std2 = np.sqrt(np.maximum(sqr2 - mean2 ** 2, 1e-8))

        # Compute correlation
        correlation = cv2.filter2D(img1 * img2, -1, kernel) - mean1 * mean2

        # Normalize with safety check
        denominator = std1 * std2
        eps = 1e-8
        ncc = np.divide(
            correlation, 
            np.maximum(denominator, eps), 
            out=np.zeros_like(correlation)
        )

        # Clip to valid range and convert to cost
        ncc = np.clip(ncc, -1, 1)
        return 1 - ncc

    def filter_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Filter depth map to remove outliers and smooth"""
        # Median filter to remove outliers
        filtered = cv2.medianBlur(depth_map.astype(np.float32), 5)

        # Bilateral filter to smooth while preserving edges
        filtered = cv2.bilateralFilter(filtered, 9, 75, 75)

        return filtered

    def fuse_depth_maps(self) -> np.ndarray:
        """Fuse multiple depth maps into a single dense point cloud"""
        print("Fusing depth maps into dense point cloud...")

        all_points = []
        all_colors = []

        K = np.array([
            [self.camera_params.focal_length, 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length, self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])

        for img_id, depth_map in self.depth_maps.items():
            if img_id not in self.camera_poses:
                continue

            pose = self.camera_poses[img_id]
            img = self.images[img_id]
            h, w = depth_map.shape

            # Create coordinate grid
            x, y = np.meshgrid(np.arange(w), np.arange(h))

            # Convert to 3D points
            valid_mask = (depth_map > 0) & (depth_map < 2.0)  # Remove invalid depths

            if not np.any(valid_mask):
                continue

            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            depth_valid = depth_map[valid_mask]

            # Backproject to 3D camera coordinates
            coords = np.stack([x_valid, y_valid, np.ones_like(x_valid)], axis=-1)
            normalized = (np.linalg.inv(K) @ coords.T).T
            points_3d = normalized * depth_valid[:, np.newaxis]

            # Transform to world coordinates
            world_points = (pose.rotation.T @ points_3d.T).T - pose.rotation.T @ pose.translation

            # Get colors
            colors = img[valid_mask] / 255.0

            all_points.extend(world_points)
            all_colors.extend(colors)

        if all_points:
            self.dense_point_cloud = np.array(all_points)
            self.dense_point_colors = np.array(all_colors)
            return self.dense_point_cloud
        else:
            return np.array([])

    def _has_dense_point_cloud(self):
        """Helper to check if dense point cloud exists and is non-empty."""
        return hasattr(self, 'dense_point_cloud') and len(self.dense_point_cloud) > 0

    def generate_mesh(self, filename: str = "mesh.ply"):
        """Generate mesh from dense point cloud using Poisson reconstruction"""
        if not hasattr(self, 'dense_point_cloud') or len(self.dense_point_cloud) == 0:
            print("No dense point cloud available for meshing")
            return None

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.dense_point_cloud)

        if hasattr(self, 'dense_point_colors'):
            pcd.colors = o3d.utility.Vector3dVector(self.dense_point_colors)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30
            )
        )

        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9, width=0, scale=1.1, linear_fit=False
        )

        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.01)
        mesh.remove_vertices_by_mask(vertices_to_remove)

        # Save mesh
        o3d.io.write_triangle_mesh(filename, mesh)
        print(f"Exported mesh to {filename}")

        return mesh

    def export_point_cloud(self, filename: str):
        """Export point cloud to PLY format"""
        if self._has_dense_point_cloud():
            # Export dense point cloud if available
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(self.dense_point_cloud)

            if hasattr(self, 'dense_point_colors'):
                point_cloud.colors = o3d.utility.Vector3dVector(self.dense_point_colors)
        else:
            # Fall back to sparse point cloud
            if not self.point_cloud:
                print("No point cloud to export")
                return

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(np.array(self.point_cloud))

            if self.point_colors:
                point_cloud.colors = o3d.utility.Vector3dVector(np.array(self.point_colors))

        o3d.io.write_point_cloud(filename, point_cloud)
        print(f"Exported point cloud to {filename}")

    def run_sfm_pipeline(self, feature_type='ORB', use_cuda=False):
        """Run the complete SfM pipeline"""
        print("Starting SfM pipeline...")
        self.load_images()
        if len(self.images) < 2:
            raise ValueError("Need at least 2 images for reconstruction")
        self.extract_features(feature_type=feature_type, use_cuda=use_cuda)
        self.match_features(feature_type=feature_type)
        self.estimate_camera_intrinsics()
        self.initialize_two_view_reconstruction()
        self.run_bundle_adjustment()
        self.export_point_cloud("reconstruction.ply")
        print("SfM pipeline completed successfully!")
        return len(self.camera_poses), len(self.point_cloud)

    def add_image_incremental(self, img_path, feature_type='ORB', use_cuda=False):
        """Add a new image incrementally, extract features, and match to existing images."""
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        self.images.append(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Feature extraction using helper method
        detector, use_cuda_detector, detector_name = self._create_detector(feature_type, use_cuda)

        if use_cuda_detector:
            gpu_img = cv2.cuda_GpuMat()
            gpu_img.upload(gray)
            kps = detector.detect(gpu_img)
            kps, descs = detector.compute(gpu_img, kps)
            descs = descs.download() if descs is not None else None
            # Release GPU memory
            del gpu_img
        else:
            kps, descs = detector.detectAndCompute(gray, None)

        self.keypoints.append(kps)
        self.descriptors.append(descs)

        # Match to all previous images with proper matcher
        matcher = self._get_matcher_for_descriptor(feature_type)
        new_idx = len(self.images) - 1
        for i in range(new_idx):
            if self.descriptors[i] is not None and descs is not None:
                raw_matches = matcher.knnMatch(self.descriptors[i], descs, k=2)
                # Apply Lowe's ratio test
                good_matches = []
                for match_pair in raw_matches:
                    if len(match_pair) == 2:
                        m, n = match_pair
                        if m.distance < 0.7 * n.distance:
                            good_matches.append(m)
                self.matches[(i, new_idx)] = good_matches

        print(f"Added image {img_path} incrementally. Features extracted and matched.")

    def incremental_sfm_update(self):
        """Update the reconstruction incrementally after adding a new image."""
        # Find the new image index
        new_idx = len(self.images) - 1
        # Find the best match to existing cameras
        best_score = 0
        best_pair = None
        for i in range(new_idx):
            matches = self.matches.get((i, new_idx), [])
            if len(matches) > best_score:
                best_score = len(matches)
                best_pair = (i, new_idx)
        if best_pair is None:
            print("No valid matches for incremental SfM update.")
            return
        # Estimate pose for the new image
        img1_idx, img2_idx = best_pair
        matches = self.matches[best_pair]
        pts1 = np.array([self.keypoints[img1_idx][m.queryIdx].pt for m in matches])
        pts2 = np.array([self.keypoints[img2_idx][m.trainIdx].pt for m in matches])
        if self.camera_params is None:
            self.estimate_camera_intrinsics()
        K = np.array([
            [self.camera_params.focal_length, 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length, self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC)
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
        # Add new camera pose
        self.camera_poses[img2_idx] = CameraPose(
            rotation=R,
            translation=t.flatten(),
            image_id=img2_idx
        )
        # Triangulate new points
        points_3d = self.triangulate_points(img1_idx, img2_idx, self.camera_poses[img1_idx], self.camera_poses[img2_idx])
        if points_3d.shape[0] > 0:
            if not hasattr(self, 'point_cloud') or self.point_cloud is None:
                self.point_cloud = []
            self.point_cloud.extend(points_3d.tolist())
        print(f"Incremental SfM update complete for image {img2_idx}. New points: {points_3d.shape[0]}")

    def run_complete_pipeline(self):
        """Run the complete photogrammetry pipeline including dense reconstruction"""
        print("Starting complete photogrammetry pipeline...")

        try:
            # Step 1-6: Basic SfM pipeline
            n_poses, n_points = self.run_sfm_pipeline()

            # Step 7: Dense reconstruction
            self.compute_depth_maps()

            # Step 8: Fuse depth maps
            dense_points = self.fuse_depth_maps()

            # Step 9: Generate mesh (optional)
            mesh = None
            if len(dense_points) > 0:
                try:
                    mesh = self.generate_mesh("reconstruction_mesh.ply")
                    if mesh is not None:
                        print(f"Generated mesh with {len(mesh.vertices)} vertices")
                except Exception as e:
                    print(f"Mesh generation failed: {e}")

            # Step 10: Export dense point cloud
            if len(dense_points) > 0:
                self.export_point_cloud("dense_reconstruction.ply")
                print(f"Generated dense point cloud with {len(dense_points)} points")

            return mesh, (dense_points if len(dense_points) > 0 else None)

        except Exception as e:
            print(f"Pipeline failed: {e}")
            return None, None


# Usage example for micro machines race track
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run photogrammetry pipeline')
    parser.add_argument('images_folder', help='Path to folder containing images')
    parser.add_argument('--feature_type', default='ORB', choices=['SIFT', 'ORB'],
                        help='Feature detector type')
    parser.add_argument('--use_cuda', action='store_true',
                        help='Use CUDA for ORB feature extraction (if available)')
    parser.add_argument('--focal_length', type=float, default=None,
                        help='Initial focal length guess')
    parser.add_argument('--dense', action='store_true',
                        help='Run dense reconstruction')

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = PhotogrammetryPipeline(args.images_folder)

        if args.dense:
            # Run complete pipeline with dense reconstruction
            mesh, dense_points = pipeline.run_complete_pipeline()

            # Print results
            if dense_points is not None:
                print(f"Generated dense point cloud with {len(dense_points)} points")
            if mesh is not None:
                print(f"Generated mesh successfully")
            print(f"Estimated {len(pipeline.camera_poses)} camera poses")

        else:
            # Run only SfM pipeline
            n_poses, n_points = pipeline.run_sfm_pipeline(feature_type=args.feature_type, use_cuda=args.use_cuda)
            print(f"SfM completed: {n_poses} poses, {n_points} sparse points")

    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

    # Optional: Visualize results (uncomment to use)
    # import open3d as o3d
    # if os.path.exists("reconstruction.ply"):
    #     pcd = o3d.io.read_point_cloud("reconstruction.ply")
    #     o3d.visualization.draw_geometries([pcd])
    # if os.path.exists("reconstruction_mesh.ply"):
    #     mesh = o3d.io.read_triangle_mesh("reconstruction_mesh.ply")
    #     o3d.visualization.draw_geometries([mesh])
