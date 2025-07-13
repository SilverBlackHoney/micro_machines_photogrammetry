import cv2
import numpy as np
import open3d as o3d
from typing import List, Tuple, Optional, Dict
from scipy.spatial import Delaunay
from scipy.ndimage import binary_erosion, binary_dilation
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class CameraPose:
    """Camera extrinsic parameters"""
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector
    image_id: int


@dataclass
class CameraParameters:
    """Camera intrinsic parameters"""
    focal_length: float
    principal_point: Tuple[float, float]
    distortion: np.ndarray
    image_size: Tuple[int, int]


class DenseReconstruction:
    """Advanced dense reconstruction methods for detailed mesh generation"""
    
    def __init__(self, camera_params: CameraParameters):
        self.camera_params = camera_params
        self.K = self._get_camera_matrix()
        self.depth_maps = {}
        self.confidence_maps = {}
        self.normal_maps = {}
        
    def _get_camera_matrix(self):
        """Get camera intrinsic matrix"""
        return np.array([
            [self.camera_params.focal_length, 0, self.camera_params.principal_point[0]],
            [0, self.camera_params.focal_length, self.camera_params.principal_point[1]],
            [0, 0, 1]
        ])
    
    def patchmatch_mvs(self, images: List[np.ndarray], poses: Dict[int, CameraPose], 
                       ref_view: int, neighbor_views: List[int],
                       patch_size: int = 7, iterations: int = 3,
                       min_depth: float = 0.1, max_depth: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        PatchMatch Multi-View Stereo for high-quality depth estimation
        
        Args:
            images: List of input images
            poses: Camera poses for each view
            ref_view: Reference view index
            neighbor_views: List of neighbor view indices
            patch_size: Size of matching patches
            iterations: Number of PatchMatch iterations
            min_depth: Minimum depth value
            max_depth: Maximum depth value
            
        Returns:
            depth_map: Estimated depth map
            confidence_map: Confidence values for each pixel
        """
        print(f"Running PatchMatch MVS for view {ref_view}")
        
        ref_img = cv2.cvtColor(images[ref_view], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        h, w = ref_img.shape
        
        # Initialize depth and normal hypotheses randomly
        depth_map = np.random.uniform(min_depth, max_depth, (h, w))
        normal_map = self._random_normal_map(h, w)
        
        # Cost computation for each hypothesis
        cost_map = np.full((h, w), float('inf'))
        
        half_patch = patch_size // 2
        
        for iteration in range(iterations):
            print(f"  PatchMatch iteration {iteration + 1}/{iterations}")
            
            # Spatial propagation
            depth_map, normal_map, cost_map = self._spatial_propagation(
                ref_img, images, poses, ref_view, neighbor_views,
                depth_map, normal_map, cost_map, patch_size, half_patch
            )
            
            # View propagation
            depth_map, normal_map, cost_map = self._view_propagation(
                ref_img, images, poses, ref_view, neighbor_views,
                depth_map, normal_map, cost_map, patch_size, half_patch
            )
            
            # Random refinement
            depth_map, normal_map, cost_map = self._random_refinement(
                ref_img, images, poses, ref_view, neighbor_views,
                depth_map, normal_map, cost_map, patch_size, half_patch,
                min_depth, max_depth
            )
        
        # Post-process depth map
        depth_map = self._post_process_depth(depth_map, ref_img)
        
        # Compute confidence based on cost consistency
        confidence_map = self._compute_confidence(cost_map)
        
        return depth_map, confidence_map
    
    def _random_normal_map(self, h: int, w: int) -> np.ndarray:
        """Generate random normal map"""
        normals = np.random.randn(h, w, 3)
        # Normalize and ensure they point towards camera (negative z)
        normals = normals / np.linalg.norm(normals, axis=2, keepdims=True)
        normals[:, :, 2] = -np.abs(normals[:, :, 2])
        return normals
    
    def _spatial_propagation(self, ref_img, images, poses, ref_view, neighbor_views,
                           depth_map, normal_map, cost_map, patch_size, half_patch):
        """Spatial propagation step of PatchMatch"""
        h, w = ref_img.shape
        new_depth = depth_map.copy()
        new_normal = normal_map.copy()
        new_cost = cost_map.copy()
        
        # Check neighbors (4-connectivity)
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        for y in range(half_patch, h - half_patch):
            for x in range(half_patch, w - half_patch):
                current_cost = cost_map[y, x]
                
                for dy, dx in directions:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        # Try neighbor's hypothesis
                        test_depth = depth_map[ny, nx]
                        test_normal = normal_map[ny, nx]
                        
                        cost = self._compute_patch_cost(
                            ref_img, images, poses, ref_view, neighbor_views,
                            x, y, test_depth, test_normal, patch_size
                        )
                        
                        if cost < current_cost:
                            new_depth[y, x] = test_depth
                            new_normal[y, x] = test_normal
                            new_cost[y, x] = cost
                            current_cost = cost
        
        return new_depth, new_normal, new_cost
    
    def _view_propagation(self, ref_img, images, poses, ref_view, neighbor_views,
                         depth_map, normal_map, cost_map, patch_size, half_patch):
        """View propagation step of PatchMatch"""
        # For simplicity, we'll skip view propagation in this implementation
        # In a full implementation, this would propagate hypotheses across views
        return depth_map, normal_map, cost_map
    
    def _random_refinement(self, ref_img, images, poses, ref_view, neighbor_views,
                          depth_map, normal_map, cost_map, patch_size, half_patch,
                          min_depth, max_depth):
        """Random refinement step of PatchMatch"""
        h, w = ref_img.shape
        new_depth = depth_map.copy()
        new_normal = normal_map.copy()
        new_cost = cost_map.copy()
        
        # Random search around current hypothesis
        search_radius = (max_depth - min_depth) * 0.5
        
        for y in range(half_patch, h - half_patch):
            for x in range(half_patch, w - half_patch):
                current_cost = cost_map[y, x]
                
                # Try random perturbations
                for _ in range(3):  # Try a few random samples
                    # Perturb depth
                    noise_depth = np.random.normal(0, search_radius * 0.1)
                    test_depth = np.clip(depth_map[y, x] + noise_depth, min_depth, max_depth)
                    
                    # Perturb normal
                    noise_normal = np.random.normal(0, 0.1, 3)
                    test_normal = normal_map[y, x] + noise_normal
                    test_normal = test_normal / np.linalg.norm(test_normal)
                    test_normal[2] = -np.abs(test_normal[2])  # Ensure pointing towards camera
                    
                    cost = self._compute_patch_cost(
                        ref_img, images, poses, ref_view, neighbor_views,
                        x, y, test_depth, test_normal, patch_size
                    )
                    
                    if cost < current_cost:
                        new_depth[y, x] = test_depth
                        new_normal[y, x] = test_normal
                        new_cost[y, x] = cost
                        current_cost = cost
                
                # Reduce search radius
                search_radius *= 0.5
        
        return new_depth, new_normal, new_cost
    
    def _compute_patch_cost(self, ref_img, images, poses, ref_view, neighbor_views,
                           x, y, depth, normal, patch_size):
        """Compute photometric cost for a patch hypothesis"""
        ref_pose = poses[ref_view]
        half_patch = patch_size // 2
        
        total_cost = 0
        valid_views = 0
        
        for neighbor_view in neighbor_views:
            if neighbor_view >= len(images):
                continue
                
            neighbor_img = cv2.cvtColor(images[neighbor_view], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            neighbor_pose = poses[neighbor_view]
            
            # Project point to neighbor view
            point_3d = self._backproject_point(x, y, depth, ref_pose)
            proj_x, proj_y = self._project_point(point_3d, neighbor_pose)
            
            if (half_patch <= proj_x < neighbor_img.shape[1] - half_patch and
                half_patch <= proj_y < neighbor_img.shape[0] - half_patch):
                
                # Extract patches
                ref_patch = ref_img[y-half_patch:y+half_patch+1, x-half_patch:x+half_patch+1]
                neighbor_patch = neighbor_img[proj_y-half_patch:proj_y+half_patch+1,
                                            proj_x-half_patch:proj_x+half_patch+1]
                
                # Compute normalized cross correlation
                if ref_patch.size > 0 and neighbor_patch.size > 0:
                    ncc = self._compute_ncc(ref_patch, neighbor_patch)
                    total_cost += (1 - ncc)
                    valid_views += 1
        
        return total_cost / max(valid_views, 1)
    
    def _backproject_point(self, x, y, depth, pose):
        """Backproject pixel to 3D world coordinates"""
        # Convert to normalized coordinates
        point_cam = np.linalg.inv(self.K) @ np.array([x, y, 1]) * depth
        # Transform to world coordinates
        point_world = pose.rotation.T @ (point_cam - pose.translation)
        return point_world
    
    def _project_point(self, point_3d, pose):
        """Project 3D point to image coordinates"""
        # Transform to camera coordinates
        point_cam = pose.rotation @ point_3d + pose.translation
        # Project to image
        if point_cam[2] > 0:
            proj = self.K @ point_cam
            return int(proj[0] / proj[2]), int(proj[1] / proj[2])
        return -1, -1
    
    def _compute_ncc(self, patch1, patch2):
        """Compute normalized cross correlation between patches"""
        patch1_flat = patch1.flatten()
        patch2_flat = patch2.flatten()
        
        if len(patch1_flat) != len(patch2_flat):
            return 0
        
        patch1_norm = patch1_flat - np.mean(patch1_flat)
        patch2_norm = patch2_flat - np.mean(patch2_flat)
        
        numerator = np.sum(patch1_norm * patch2_norm)
        denominator = np.sqrt(np.sum(patch1_norm**2) * np.sum(patch2_norm**2))
        
        if denominator > 0:
            return numerator / denominator
        return 0
    
    def _post_process_depth(self, depth_map, ref_img):
        """Post-process depth map with edge-preserving filtering"""
        # Bilateral filtering to preserve edges while smoothing
        depth_8bit = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
        filtered = cv2.bilateralFilter(depth_8bit, 9, 75, 75)
        return filtered.astype(np.float32) / 255.0 * (depth_map.max() - depth_map.min()) + depth_map.min()
    
    def _compute_confidence(self, cost_map):
        """Compute confidence map from cost values"""
        # Invert costs so lower cost = higher confidence
        max_cost = np.percentile(cost_map[cost_map != float('inf')], 95)
        confidence = np.clip(1.0 - cost_map / max_cost, 0, 1)
        confidence[cost_map == float('inf')] = 0
        return confidence
    
    def semi_global_matching(self, img1: np.ndarray, img2: np.ndarray, 
                           pose1: CameraPose, pose2: CameraPose,
                           max_disparity: int = 128, p1: int = 8, p2: int = 32) -> np.ndarray:
        """
        Semi-Global Matching for robust stereo depth estimation
        
        Args:
            img1, img2: Stereo image pair
            pose1, pose2: Camera poses
            max_disparity: Maximum disparity search range
            p1, p2: SGM penalty parameters
            
        Returns:
            depth_map: Estimated depth map
        """
        print("Running Semi-Global Matching")
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Create SGM matcher
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=max_disparity,
            blockSize=5,
            P1=p1 * 5**2,
            P2=p2 * 5**2,
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=0,
            speckleRange=2,
            preFilterCap=63,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        
        # Compute disparity
        disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0
        
        # Convert disparity to depth
        baseline = np.linalg.norm(pose2.translation - pose1.translation)
        depth_map = (self.camera_params.focal_length * baseline) / (disparity + 1e-10)
        
        # Filter invalid depths
        depth_map[disparity <= 0] = 0
        depth_map[depth_map > 100] = 0  # Remove very far points
        
        return depth_map
    
    def tsdf_fusion(self, depth_maps: Dict[int, np.ndarray], poses: Dict[int, CameraPose],
                    images: List[np.ndarray], voxel_size: float = 0.01,
                    truncation_distance: float = 0.04) -> o3d.geometry.TriangleMesh:
        """
        TSDF (Truncated Signed Distance Function) volume fusion for robust mesh generation
        
        Args:
            depth_maps: Dictionary of depth maps for each view
            poses: Camera poses
            images: RGB images
            voxel_size: Size of voxels in the TSDF volume
            truncation_distance: TSDF truncation distance
            
        Returns:
            mesh: Fused triangle mesh
        """
        print("Running TSDF volume fusion")
        
        # Initialize TSDF volume
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=truncation_distance,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        
        # Integrate depth maps
        for view_id, depth_map in depth_maps.items():
            if view_id not in poses or view_id >= len(images):
                continue
                
            print(f"  Integrating view {view_id}")
            
            # Convert to Open3D format
            rgb = o3d.geometry.Image(images[view_id])
            depth = o3d.geometry.Image((depth_map * 1000).astype(np.uint16))  # Convert to mm
            
            # Create RGBD image
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                rgb, depth, depth_scale=1000.0, depth_trunc=truncation_distance * 1000
            )
            
            # Camera intrinsics
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                width=self.camera_params.image_size[0],
                height=self.camera_params.image_size[1],
                fx=self.camera_params.focal_length,
                fy=self.camera_params.focal_length,
                cx=self.camera_params.principal_point[0],
                cy=self.camera_params.principal_point[1]
            )
            
            # Camera extrinsics
            pose = poses[view_id]
            extrinsic = np.eye(4)
            extrinsic[:3, :3] = pose.rotation
            extrinsic[:3, 3] = pose.translation
            
            # Integrate into volume
            volume.integrate(rgbd, intrinsic, extrinsic)
        
        # Extract triangle mesh
        print("Extracting mesh from TSDF volume")
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        
        return mesh
    
    def delaunay_triangulation_mesh(self, points_3d: np.ndarray, colors: np.ndarray = None,
                                   poses: Dict[int, CameraPose] = None) -> o3d.geometry.TriangleMesh:
        """
        Generate mesh using Delaunay triangulation with visibility constraints
        
        Args:
            points_3d: 3D point cloud
            colors: Point colors (optional)
            poses: Camera poses for visibility checking (optional)
            
        Returns:
            mesh: Triangle mesh
        """
        print("Generating mesh using Delaunay triangulation")
        
        if len(points_3d) < 4:
            print("Not enough points for triangulation")
            return o3d.geometry.TriangleMesh()
        
        # Project points to 2D for triangulation (use XY plane)
        points_2d = points_3d[:, :2]
        
        try:
            # Perform Delaunay triangulation
            tri = Delaunay(points_2d)
            
            # Create mesh
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(points_3d)
            mesh.triangles = o3d.utility.Vector3iVector(tri.simplices)
            
            if colors is not None:
                mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
            
            # Filter triangles based on visibility if poses provided
            if poses is not None and len(poses) > 0:
                mesh = self._filter_triangles_by_visibility(mesh, poses)
            
            # Remove degenerate triangles
            mesh.remove_degenerate_triangles()
            mesh.remove_duplicated_triangles()
            mesh.remove_duplicated_vertices()
            mesh.remove_non_manifold_edges()
            
            mesh.compute_vertex_normals()
            
            return mesh
            
        except Exception as e:
            print(f"Delaunay triangulation failed: {e}")
            return o3d.geometry.TriangleMesh()
    
    def _filter_triangles_by_visibility(self, mesh, poses):
        """Filter triangles based on camera visibility"""
        # Simplified visibility filtering - remove triangles that are not visible
        # from any camera
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        
        visible_triangles = []
        
        for tri_idx, triangle in enumerate(triangles):
            tri_center = np.mean(vertices[triangle], axis=0)
            
            # Check if triangle center is visible from any camera
            visible = False
            for pose in poses.values():
                # Transform to camera coordinates
                point_cam = pose.rotation @ tri_center + pose.translation
                
                # Check if in front of camera
                if point_cam[2] > 0:
                    # Project to image
                    proj = self.K @ point_cam
                    x, y = proj[0] / proj[2], proj[1] / proj[2]
                    
                    # Check if within image bounds
                    if (0 <= x < self.camera_params.image_size[0] and
                        0 <= y < self.camera_params.image_size[1]):
                        visible = True
                        break
            
            if visible:
                visible_triangles.append(triangle)
        
        if visible_triangles:
            mesh.triangles = o3d.utility.Vector3iVector(np.array(visible_triangles))
        
        return mesh
    
    def advanced_mesh_processing(self, mesh: o3d.geometry.TriangleMesh,
                                iterations: int = 2) -> o3d.geometry.TriangleMesh:
        """
        Advanced mesh post-processing for high-quality results
        
        Args:
            mesh: Input triangle mesh
            iterations: Number of processing iterations
            
        Returns:
            processed_mesh: Processed mesh
        """
        print("Applying advanced mesh processing")
        
        processed_mesh = mesh
        
        for i in range(iterations):
            print(f"  Processing iteration {i + 1}/{iterations}")
            
            # Remove statistical outliers
            processed_mesh, _ = processed_mesh.remove_statistical_outlier(
                nb_neighbors=20, std_ratio=2.0
            )
            
            # Smooth mesh while preserving features
            processed_mesh = processed_mesh.filter_smooth_taubin(
                number_of_iterations=10, lambda_filter=0.5, mu=-0.53
            )
            
            # Simplify mesh if too dense
            if len(processed_mesh.vertices) > 100000:
                target_triangles = min(50000, len(processed_mesh.triangles) // 2)
                processed_mesh = processed_mesh.simplify_quadric_decimation(target_triangles)
            
            # Fill holes
            if hasattr(processed_mesh, 'fill_holes'):
                processed_mesh.fill_holes(hole_size=1000)
            
            # Recompute normals
            processed_mesh.compute_vertex_normals()
        
        # Final cleanup
        processed_mesh.remove_degenerate_triangles()
        processed_mesh.remove_duplicated_triangles()
        processed_mesh.remove_duplicated_vertices()
        processed_mesh.remove_non_manifold_edges()
        
        return processed_mesh
    
    def multi_resolution_fusion(self, depth_maps: Dict[int, np.ndarray],
                               confidence_maps: Dict[int, np.ndarray],
                               poses: Dict[int, CameraPose]) -> np.ndarray:
        """
        Multi-resolution depth map fusion for detailed reconstruction
        
        Args:
            depth_maps: Dictionary of depth maps
            confidence_maps: Dictionary of confidence maps
            poses: Camera poses
            
        Returns:
            fused_points: Fused 3D point cloud
        """
        print("Running multi-resolution depth fusion")
        
        all_points = []
        all_colors = []
        all_confidences = []
        
        for view_id, depth_map in depth_maps.items():
            if view_id not in poses or view_id not in confidence_maps:
                continue
                
            confidence_map = confidence_maps[view_id]
            pose = poses[view_id]
            
            h, w = depth_map.shape
            
            # Create coordinate grids
            x, y = np.meshgrid(np.arange(w), np.arange(h))
            
            # Filter by confidence
            valid_mask = (depth_map > 0) & (confidence_map > 0.5)
            
            if not np.any(valid_mask):
                continue
            
            x_valid = x[valid_mask]
            y_valid = y[valid_mask]
            depth_valid = depth_map[valid_mask]
            confidence_valid = confidence_map[valid_mask]
            
            # Backproject to 3D
            points_3d = []
            for i in range(len(x_valid)):
                point_3d = self._backproject_point(x_valid[i], y_valid[i], depth_valid[i], pose)
                points_3d.append(point_3d)
            
            if points_3d:
                all_points.extend(points_3d)
                all_confidences.extend(confidence_valid)
        
        if all_points:
            points_array = np.array(all_points)
            confidences_array = np.array(all_confidences)
            
            # Weighted fusion based on confidence
            fused_points = self._confidence_weighted_fusion(points_array, confidences_array)
            
            return fused_points
        
        return np.array([])
    
    def _confidence_weighted_fusion(self, points, confidences, tolerance=0.02):
        """Fuse nearby points based on confidence weights"""
        if len(points) == 0:
            return points
        
        # Simple clustering based on spatial proximity
        fused_points = []
        used = np.zeros(len(points), dtype=bool)
        
        for i, point in enumerate(points):
            if used[i]:
                continue
                
            # Find nearby points
            distances = np.linalg.norm(points - point, axis=1)
            nearby_mask = distances < tolerance
            nearby_indices = np.where(nearby_mask)[0]
            
            if len(nearby_indices) > 1:
                # Weighted average based on confidence
                nearby_points = points[nearby_indices]
                nearby_confidences = confidences[nearby_indices]
                
                weights = nearby_confidences / np.sum(nearby_confidences)
                fused_point = np.sum(nearby_points * weights[:, np.newaxis], axis=0)
                
                fused_points.append(fused_point)
                used[nearby_indices] = True
            else:
                fused_points.append(point)
                used[i] = True
        
        return np.array(fused_points)