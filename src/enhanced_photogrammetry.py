import cv2
import numpy as np
from scipy.optimize import least_squares
import open3d as o3d
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import os
import matplotlib.pyplot as plt

from dense_reconstruction import DenseReconstruction, CameraParameters, CameraPose


class EnhancedPhotogrammetryPipeline:
    """Enhanced photogrammetry pipeline with advanced dense reconstruction methods"""
    
    def __init__(self, images_folder: str):
        if not os.path.exists(images_folder):
            raise ValueError(f"Images folder does not exist: {images_folder}")
        
        self.images_folder = images_folder
        self.images = []
        self.keypoints = []
        self.descriptors = []
        self.matches = {}
        self.camera_params = None
        self.camera_poses = {}
        self.point_cloud = []
        self.point_colors = []
        self.track_graph = {}
        
        # Dense reconstruction components
        self.dense_reconstructor = None
        self.dense_depth_maps = {}
        self.confidence_maps = {}
        self.dense_point_cloud = None
        self.dense_point_colors = None
        
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

    def extract_features(self, feature_type='SIFT'):
        """Extract features from all images"""
        if feature_type == 'SIFT':
            if hasattr(cv2, 'SIFT_create'):
                detector = cv2.SIFT_create(nfeatures=2000)
            else:
                raise ValueError(
                    "SIFT is not available in your OpenCV installation. "
                    "Please install opencv-contrib-python."
                )
        elif feature_type == 'ORB':
            if hasattr(cv2, 'ORB_create'):
                detector = cv2.ORB_create(nfeatures=2000)
            else:
                raise ValueError(
                    "ORB is not available in your OpenCV installation."
                )
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        for img in self.images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kps, descs = detector.detectAndCompute(gray, None)
            self.keypoints.append(kps)
            self.descriptors.append(descs)

        print(f"Extracted features from {len(self.images)} images")

    def match_features(self, ratio_threshold=0.7):
        """Match features between all image pairs"""
        matcher = cv2.BFMatcher()

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

                    self.matches[(i, j)] = good_matches

        print(f"Matched features for {len(self.matches)} image pairs")

    def estimate_camera_intrinsics(self, focal_length_guess=None):
        """Estimate camera intrinsic parameters"""
        if len(self.images) == 0:
            raise ValueError("No images loaded")

        h, w = self.images[0].shape[:2]
        
        if focal_length_guess is None:
            focal_length = max(w, h)  # Rough estimate
        else:
            focal_length = focal_length_guess

        principal_point = (w / 2, h / 2)
        distortion = np.zeros(5)

        self.camera_params = CameraParameters(
            focal_length=focal_length,
            principal_point=principal_point,
            distortion=distortion,
            image_size=(w, h)
        )
        
        # Initialize dense reconstructor
        self.dense_reconstructor = DenseReconstruction(self.camera_params)

        return self.camera_params

    def estimate_fundamental_matrix(self, img1_idx: int, img2_idx: int) -> np.ndarray:
        """Estimate fundamental matrix between two images"""
        matches = self.matches.get((img1_idx, img2_idx), [])
        if len(matches) < 8:
            return None

        # Extract matching points
        pts1 = np.array([self.keypoints[img1_idx][m.queryIdx].pt for m in matches])
        pts2 = np.array([self.keypoints[img2_idx][m.trainIdx].pt for m in matches])

        # Estimate fundamental matrix using RANSAC
        F, mask = cv2.findFundamentalMat(
            pts1, pts2, cv2.FM_RANSAC, 1.0, 0.99
        )

        return F

    def triangulate_points(
        self,
        img1_idx: int,
        img2_idx: int,
        pose1: CameraPose,
        pose2: CameraPose
    ) -> np.ndarray:
        """Triangulate 3D points from two views"""
        matches = self.matches.get((img1_idx, img2_idx), [])
        if len(matches) == 0:
            return np.array([])

        # Camera matrices
        K = np.array([
            [self.camera_params.focal_length, 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length, self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])

        # Projection matrices
        P1 = K @ np.hstack([pose1.rotation, pose1.translation.reshape(-1, 1)])
        P2 = K @ np.hstack([pose2.rotation, pose2.translation.reshape(-1, 1)])

        # Extract matching points
        pts1 = np.array([self.keypoints[img1_idx][m.queryIdx].pt for m in matches])
        pts2 = np.array([self.keypoints[img2_idx][m.trainIdx].pt for m in matches])

        # Triangulate points
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        points_3d = points_4d[:3] / points_4d[3]

        return points_3d.T

    def initialize_two_view_reconstruction(self):
        """Initialize reconstruction from two views with most matches"""
        best_pair = None
        max_matches = 0

        for pair, matches in self.matches.items():
            if len(matches) > max_matches:
                max_matches = len(matches)
                best_pair = pair

        if best_pair is None or max_matches < 100:
            raise ValueError("Insufficient matches for initialization")

        img1_idx, img2_idx = best_pair
        matches = self.matches[best_pair]

        print(f"Initializing reconstruction with images {img1_idx} and {img2_idx}")
        print(f"Number of matches: {len(matches)}")

        # Extract matching points
        pts1 = np.array([self.keypoints[img1_idx][m.queryIdx].pt for m in matches])
        pts2 = np.array([self.keypoints[img2_idx][m.trainIdx].pt for m in matches])

        # Estimate essential matrix
        K = np.array([
            [self.camera_params.focal_length, 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length, self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])

        E, mask = cv2.findEssentialMat(
            pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0
        )

        # Recover pose
        _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)

        # Set camera poses
        self.camera_poses[img1_idx] = CameraPose(
            rotation=np.eye(3),
            translation=np.zeros(3),
            image_id=img1_idx
        )

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

        # Filter points by reprojection error and depth
        valid_points = []
        valid_colors = []
        
        for i, point in enumerate(points_3d):
            if point[2] > 0:  # Positive depth
                valid_points.append(point)
                # Get color from first image
                match = matches[i] if i < len(matches) else matches[0]
                kp = self.keypoints[img1_idx][match.queryIdx]
                x, y = int(kp.pt[0]), int(kp.pt[1])
                if 0 <= x < self.images[img1_idx].shape[1] and 0 <= y < self.images[img1_idx].shape[0]:
                    color = self.images[img1_idx][y, x] / 255.0
                    valid_colors.append(color)
                else:
                    valid_colors.append([0.5, 0.5, 0.5])

        self.point_cloud = np.array(valid_points)
        self.point_colors = np.array(valid_colors)

        print(f"Initialized with {len(self.point_cloud)} 3D points")

        return len(self.camera_poses), len(self.point_cloud)

    def run_dense_reconstruction(self, method='patchmatch', **kwargs):
        """
        Run dense reconstruction using specified method
        
        Args:
            method: 'patchmatch', 'sgm', 'tsdf', or 'delaunay'
            **kwargs: Method-specific parameters
        """
        if self.dense_reconstructor is None:
            raise ValueError("Camera parameters not estimated. Run estimate_camera_intrinsics() first.")
        
        if len(self.camera_poses) < 2:
            raise ValueError("Insufficient camera poses. Run SfM pipeline first.")
        
        print(f"Running dense reconstruction using method: {method}")
        
        if method == 'patchmatch':
            self._run_patchmatch_reconstruction(**kwargs)
        elif method == 'sgm':
            self._run_sgm_reconstruction(**kwargs)
        elif method == 'tsdf':
            self._run_tsdf_reconstruction(**kwargs)
        elif method == 'delaunay':
            self._run_delaunay_reconstruction(**kwargs)
        else:
            raise ValueError(f"Unknown reconstruction method: {method}")

    def _run_patchmatch_reconstruction(self, patch_size=7, iterations=3, 
                                     min_depth=0.1, max_depth=5.0):
        """Run PatchMatch MVS reconstruction"""
        print("Running PatchMatch Multi-View Stereo reconstruction")
        
        for img_id in self.camera_poses.keys():
            # Find neighboring views
            neighbor_views = self._find_neighboring_views(img_id, max_neighbors=4)
            
            if len(neighbor_views) > 0:
                print(f"Computing depth map for view {img_id}")
                depth_map, confidence_map = self.dense_reconstructor.patchmatch_mvs(
                    self.images, self.camera_poses, img_id, neighbor_views,
                    patch_size=patch_size, iterations=iterations,
                    min_depth=min_depth, max_depth=max_depth
                )
                
                self.dense_depth_maps[img_id] = depth_map
                self.confidence_maps[img_id] = confidence_map
        
        # Fuse depth maps into point cloud
        self._fuse_dense_depth_maps()

    def _run_sgm_reconstruction(self, max_disparity=128, p1=8, p2=32):
        """Run Semi-Global Matching reconstruction"""
        print("Running Semi-Global Matching reconstruction")
        
        # Use stereo pairs for SGM
        camera_ids = list(self.camera_poses.keys())
        
        for i in range(len(camera_ids) - 1):
            img1_id = camera_ids[i]
            img2_id = camera_ids[i + 1]
            
            print(f"Computing SGM depth for pair ({img1_id}, {img2_id})")
            
            depth_map = self.dense_reconstructor.semi_global_matching(
                self.images[img1_id], self.images[img2_id],
                self.camera_poses[img1_id], self.camera_poses[img2_id],
                max_disparity=max_disparity, p1=p1, p2=p2
            )
            
            self.dense_depth_maps[img1_id] = depth_map
            # Simple confidence based on depth consistency
            self.confidence_maps[img1_id] = (depth_map > 0).astype(np.float32)
        
        self._fuse_dense_depth_maps()

    def _run_tsdf_reconstruction(self, voxel_size=0.01, truncation_distance=0.04):
        """Run TSDF volume fusion reconstruction"""
        if len(self.dense_depth_maps) == 0:
            raise ValueError("No depth maps available. Run depth estimation first.")
        
        print("Running TSDF volume fusion")
        
        mesh = self.dense_reconstructor.tsdf_fusion(
            self.dense_depth_maps, self.camera_poses, self.images,
            voxel_size=voxel_size, truncation_distance=truncation_distance
        )
        
        return mesh

    def _run_delaunay_reconstruction(self):
        """Run Delaunay triangulation reconstruction"""
        if self.dense_point_cloud is None or len(self.dense_point_cloud) == 0:
            raise ValueError("No dense point cloud available. Run depth estimation first.")
        
        print("Running Delaunay triangulation reconstruction")
        
        mesh = self.dense_reconstructor.delaunay_triangulation_mesh(
            self.dense_point_cloud, self.dense_point_colors, self.camera_poses
        )
        
        return mesh

    def _find_neighboring_views(self, ref_view: int, max_neighbors: int = 5) -> List[int]:
        """Find neighboring views for multi-view stereo"""
        if ref_view not in self.camera_poses:
            return []
        
        ref_pose = self.camera_poses[ref_view]
        neighbors = []
        
        for view_id, pose in self.camera_poses.items():
            if view_id == ref_view:
                continue
            
            # Compute baseline distance
            baseline = np.linalg.norm(pose.translation - ref_pose.translation)
            
            # Compute viewing angle similarity
            view_angle = np.arccos(np.clip(
                np.trace(ref_pose.rotation.T @ pose.rotation) - 1, -1, 1
            )) / 2
            
            # Score based on baseline and viewing angle
            score = baseline * (1 + np.cos(view_angle))
            neighbors.append((view_id, score))
        
        # Sort by score and return top neighbors
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [view_id for view_id, _ in neighbors[:max_neighbors]]

    def _fuse_dense_depth_maps(self):
        """Fuse dense depth maps into point cloud"""
        if self.dense_reconstructor is None:
            return
        
        print("Fusing dense depth maps into point cloud")
        
        if len(self.confidence_maps) > 0:
            # Use multi-resolution fusion if confidence maps available
            fused_points = self.dense_reconstructor.multi_resolution_fusion(
                self.dense_depth_maps, self.confidence_maps, self.camera_poses
            )
        else:
            # Simple fusion
            all_points = []
            all_colors = []
            
            K = self.dense_reconstructor.K
            
            for view_id, depth_map in self.dense_depth_maps.items():
                if view_id not in self.camera_poses:
                    continue
                
                pose = self.camera_poses[view_id]
                img = self.images[view_id]
                h, w = depth_map.shape
                
                # Create coordinate grid
                x, y = np.meshgrid(np.arange(w), np.arange(h))
                valid_mask = depth_map > 0
                
                if not np.any(valid_mask):
                    continue
                
                x_valid = x[valid_mask]
                y_valid = y[valid_mask]
                depth_valid = depth_map[valid_mask]
                
                # Backproject to 3D
                for i in range(len(x_valid)):
                    point_3d = self.dense_reconstructor._backproject_point(
                        x_valid[i], y_valid[i], depth_valid[i], pose
                    )
                    all_points.append(point_3d)
                    
                    # Get color
                    color = img[y_valid[i], x_valid[i]] / 255.0
                    all_colors.append(color)
            
            fused_points = np.array(all_points) if all_points else np.array([])
            all_colors = np.array(all_colors) if all_colors else np.array([])
        
        self.dense_point_cloud = fused_points
        if len(all_colors) > 0:
            self.dense_point_colors = all_colors

    def generate_detailed_mesh(self, method='tsdf', post_process=True, **kwargs):
        """
        Generate detailed mesh using specified method
        
        Args:
            method: 'tsdf', 'delaunay', or 'poisson'
            post_process: Whether to apply advanced post-processing
            **kwargs: Method-specific parameters
        """
        print(f"Generating detailed mesh using method: {method}")
        
        if method == 'tsdf':
            mesh = self._run_tsdf_reconstruction(**kwargs)
        elif method == 'delaunay':
            mesh = self._run_delaunay_reconstruction()
        elif method == 'poisson':
            mesh = self._generate_poisson_mesh(**kwargs)
        else:
            raise ValueError(f"Unknown meshing method: {method}")
        
        if mesh is None or len(mesh.vertices) == 0:
            print("Failed to generate mesh")
            return None
        
        # Apply advanced post-processing
        if post_process and self.dense_reconstructor is not None:
            print("Applying advanced mesh post-processing")
            mesh = self.dense_reconstructor.advanced_mesh_processing(mesh)
        
        return mesh

    def _generate_poisson_mesh(self, depth=9, density_threshold=0.01):
        """Generate mesh using Poisson reconstruction"""
        if self.dense_point_cloud is None or len(self.dense_point_cloud) == 0:
            raise ValueError("No dense point cloud available")
        
        print("Running Poisson surface reconstruction")
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.dense_point_cloud)
        
        if self.dense_point_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.dense_point_colors)
        
        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )
        
        # Remove low density vertices
        if len(densities) > 0:
            vertices_to_remove = densities < np.quantile(densities, density_threshold)
            mesh.remove_vertices_by_mask(vertices_to_remove)
        
        return mesh

    def run_complete_dense_pipeline(self, reconstruction_method='patchmatch', 
                                   meshing_method='tsdf', output_dir='output'):
        """
        Run complete dense reconstruction pipeline
        
        Args:
            reconstruction_method: 'patchmatch', 'sgm'
            meshing_method: 'tsdf', 'delaunay', 'poisson'
            output_dir: Directory to save results
        """
        print("=" * 50)
        print("ENHANCED DENSE RECONSTRUCTION PIPELINE")
        print("=" * 50)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Step 1: Load images and extract features
            print("\n1. Loading images and extracting features...")
            self.load_images()
            self.extract_features()
            self.match_features()
            
            # Step 2: Estimate camera parameters
            print("\n2. Estimating camera parameters...")
            self.estimate_camera_intrinsics()
            
            # Step 3: Initialize reconstruction
            print("\n3. Initializing reconstruction...")
            self.initialize_two_view_reconstruction()
            
            # Step 4: Dense reconstruction
            print(f"\n4. Running dense reconstruction ({reconstruction_method})...")
            self.run_dense_reconstruction(method=reconstruction_method)
            
            # Step 5: Generate detailed mesh
            print(f"\n5. Generating detailed mesh ({meshing_method})...")
            mesh = self.generate_detailed_mesh(method=meshing_method)
            
            if mesh is not None:
                # Save results
                mesh_file = os.path.join(output_dir, f"detailed_mesh_{meshing_method}.ply")
                o3d.io.write_triangle_mesh(mesh_file, mesh)
                print(f"Detailed mesh saved to: {mesh_file}")
                
                # Save dense point cloud if available
                if self.dense_point_cloud is not None and len(self.dense_point_cloud) > 0:
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(self.dense_point_cloud)
                    if self.dense_point_colors is not None:
                        pcd.colors = o3d.utility.Vector3dVector(self.dense_point_colors)
                    
                    pcd_file = os.path.join(output_dir, "dense_point_cloud.ply")
                    o3d.io.write_point_cloud(pcd_file, pcd)
                    print(f"Dense point cloud saved to: {pcd_file}")
                
                # Generate visualization
                self._save_reconstruction_summary(output_dir, mesh)
                
                print(f"\n✓ Dense reconstruction completed successfully!")
                print(f"  - Mesh vertices: {len(mesh.vertices)}")
                print(f"  - Mesh triangles: {len(mesh.triangles)}")
                if self.dense_point_cloud is not None:
                    print(f"  - Dense points: {len(self.dense_point_cloud)}")
                
                return mesh
            else:
                print("\n✗ Failed to generate detailed mesh")
                return None
                
        except Exception as e:
            print(f"\n✗ Pipeline failed with error: {e}")
            return None

    def _save_reconstruction_summary(self, output_dir: str, mesh):
        """Save reconstruction summary and visualizations"""
        try:
            # Create summary plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot 1: Number of features per image
            if self.keypoints:
                feature_counts = [len(kp) for kp in self.keypoints]
                axes[0, 0].bar(range(len(feature_counts)), feature_counts)
                axes[0, 0].set_title('Features per Image')
                axes[0, 0].set_xlabel('Image Index')
                axes[0, 0].set_ylabel('Number of Features')
            
            # Plot 2: Camera poses
            if self.camera_poses:
                positions = np.array([pose.translation for pose in self.camera_poses.values()])
                if len(positions) > 0:
                    axes[0, 1].scatter(positions[:, 0], positions[:, 1])
                    axes[0, 1].set_title('Camera Positions (Top View)')
                    axes[0, 1].set_xlabel('X')
                    axes[0, 1].set_ylabel('Y')
                    axes[0, 1].axis('equal')
            
            # Plot 3: Dense point cloud
            if self.dense_point_cloud is not None and len(self.dense_point_cloud) > 0:
                # Sample points for visualization
                n_sample = min(5000, len(self.dense_point_cloud))
                indices = np.random.choice(len(self.dense_point_cloud), n_sample, replace=False)
                sample_points = self.dense_point_cloud[indices]
                
                axes[1, 0].scatter(sample_points[:, 0], sample_points[:, 2], s=1)
                axes[1, 0].set_title('Dense Point Cloud (Side View)')
                axes[1, 0].set_xlabel('X')
                axes[1, 0].set_ylabel('Z')
            
            # Plot 4: Mesh statistics
            if mesh is not None:
                mesh_stats = [
                    len(mesh.vertices),
                    len(mesh.triangles),
                    len(self.dense_point_cloud) if self.dense_point_cloud is not None else 0
                ]
                labels = ['Vertices', 'Triangles', 'Dense Points']
                axes[1, 1].bar(labels, mesh_stats)
                axes[1, 1].set_title('Reconstruction Statistics')
                axes[1, 1].set_ylabel('Count')
            
            plt.tight_layout()
            summary_file = os.path.join(output_dir, 'reconstruction_summary.png')
            plt.savefig(summary_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Reconstruction summary saved to: {summary_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save reconstruction summary: {e}")

    def export_dense_point_cloud(self, filename: str):
        """Export dense point cloud to file"""
        if self.dense_point_cloud is None or len(self.dense_point_cloud) == 0:
            print("No dense point cloud available for export")
            return
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.dense_point_cloud)
        
        if self.dense_point_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(self.dense_point_colors)
        
        o3d.io.write_point_cloud(filename, pcd)
        print(f"Dense point cloud exported to {filename}")