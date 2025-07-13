#!/usr/bin/env python3
"""
Demonstration script for enhanced dense reconstruction methods.

This script shows how to use the new dense reconstruction capabilities
to generate detailed meshes from image sequences.
"""

import os
import sys

# Add src directory to path if running from project root
if os.path.exists('src'):
    sys.path.append('src')

from enhanced_photogrammetry import EnhancedPhotogrammetryPipeline


def demo_patchmatch_reconstruction(images_folder: str, output_dir: str = 'output_patchmatch'):
    """
    Demonstrate PatchMatch Multi-View Stereo reconstruction
    
    Args:
        images_folder: Path to folder containing input images
        output_dir: Output directory for results
    """
    print("=" * 60)
    print("DEMO: PatchMatch Multi-View Stereo Reconstruction")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = EnhancedPhotogrammetryPipeline(images_folder)
    
    # Run complete dense reconstruction pipeline
    mesh = pipeline.run_complete_dense_pipeline(
        reconstruction_method='patchmatch',
        meshing_method='tsdf',
        output_dir=output_dir
    )
    
    if mesh is not None:
        print(f"\n✓ PatchMatch reconstruction completed!")
        print(f"  Output saved to: {output_dir}")
        return True
    else:
        print(f"\n✗ PatchMatch reconstruction failed!")
        return False


def demo_sgm_reconstruction(images_folder: str, output_dir: str = 'output_sgm'):
    """
    Demonstrate Semi-Global Matching reconstruction
    
    Args:
        images_folder: Path to folder containing input images
        output_dir: Output directory for results
    """
    print("=" * 60)
    print("DEMO: Semi-Global Matching Reconstruction")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = EnhancedPhotogrammetryPipeline(images_folder)
    
    # Run complete dense reconstruction pipeline
    mesh = pipeline.run_complete_dense_pipeline(
        reconstruction_method='sgm',
        meshing_method='poisson',
        output_dir=output_dir
    )
    
    if mesh is not None:
        print(f"\n✓ SGM reconstruction completed!")
        print(f"  Output saved to: {output_dir}")
        return True
    else:
        print(f"\n✗ SGM reconstruction failed!")
        return False


def demo_comparison(images_folder: str):
    """
    Demonstrate comparison of different reconstruction methods
    
    Args:
        images_folder: Path to folder containing input images
    """
    print("=" * 60)
    print("DEMO: Comparison of Dense Reconstruction Methods")
    print("=" * 60)
    
    methods = [
        ('patchmatch', 'tsdf', 'output_comparison_patchmatch_tsdf'),
        ('patchmatch', 'poisson', 'output_comparison_patchmatch_poisson'),
        ('sgm', 'tsdf', 'output_comparison_sgm_tsdf'),
        ('sgm', 'delaunay', 'output_comparison_sgm_delaunay'),
    ]
    
    results = {}
    
    for depth_method, mesh_method, output_dir in methods:
        print(f"\nTesting {depth_method} + {mesh_method}...")
        
        try:
            pipeline = EnhancedPhotogrammetryPipeline(images_folder)
            mesh = pipeline.run_complete_dense_pipeline(
                reconstruction_method=depth_method,
                meshing_method=mesh_method,
                output_dir=output_dir
            )
            
            if mesh is not None:
                results[f"{depth_method}+{mesh_method}"] = {
                    'success': True,
                    'vertices': len(mesh.vertices),
                    'triangles': len(mesh.triangles),
                    'output_dir': output_dir
                }
                print(f"  ✓ Success: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
            else:
                results[f"{depth_method}+{mesh_method}"] = {'success': False}
                print(f"  ✗ Failed")
                
        except Exception as e:
            results[f"{depth_method}+{mesh_method}"] = {'success': False, 'error': str(e)}
            print(f"  ✗ Error: {e}")
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("RECONSTRUCTION COMPARISON SUMMARY")
    print("=" * 60)
    
    for method, result in results.items():
        if result['success']:
            print(f"✓ {method:25} - {result['vertices']:6d} vertices, {result['triangles']:6d} triangles")
        else:
            error_msg = result.get('error', 'Unknown error')
            print(f"✗ {method:25} - Failed: {error_msg}")


def demo_advanced_features(images_folder: str, output_dir: str = 'output_advanced'):
    """
    Demonstrate advanced dense reconstruction features
    
    Args:
        images_folder: Path to folder containing input images
        output_dir: Output directory for results
    """
    print("=" * 60)
    print("DEMO: Advanced Dense Reconstruction Features")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = EnhancedPhotogrammetryPipeline(images_folder)
    
    try:
        # Step 1: Basic setup
        print("\n1. Setting up pipeline...")
        pipeline.load_images()
        pipeline.extract_features()
        pipeline.match_features()
        pipeline.estimate_camera_intrinsics()
        pipeline.initialize_two_view_reconstruction()
        
        # Step 2: Run PatchMatch with custom parameters
        print("\n2. Running PatchMatch with custom parameters...")
        pipeline.run_dense_reconstruction(
            method='patchmatch',
            patch_size=9,          # Larger patches for better matching
            iterations=5,          # More iterations for better quality
            min_depth=0.05,        # Closer minimum depth
            max_depth=10.0         # Further maximum depth
        )
        
        # Step 3: Generate multiple mesh types
        print("\n3. Generating meshes with different methods...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # TSDF mesh with fine voxels
        tsdf_mesh = pipeline.generate_detailed_mesh(
            method='tsdf',
            voxel_size=0.005,           # Fine voxel size
            truncation_distance=0.02,   # Small truncation
            post_process=True
        )
        if tsdf_mesh:
            tsdf_file = os.path.join(output_dir, "advanced_tsdf_mesh.ply")
            import open3d as o3d
            o3d.io.write_triangle_mesh(tsdf_file, tsdf_mesh)
            print(f"  TSDF mesh saved: {tsdf_file}")
        
        # Poisson mesh with high detail
        poisson_mesh = pipeline.generate_detailed_mesh(
            method='poisson',
            depth=10,                   # High depth for detail
            density_threshold=0.005,    # Keep more vertices
            post_process=True
        )
        if poisson_mesh:
            poisson_file = os.path.join(output_dir, "advanced_poisson_mesh.ply")
            o3d.io.write_triangle_mesh(poisson_file, poisson_mesh)
            print(f"  Poisson mesh saved: {poisson_file}")
        
        # Delaunay mesh with visibility filtering
        delaunay_mesh = pipeline.generate_detailed_mesh(
            method='delaunay',
            post_process=True
        )
        if delaunay_mesh:
            delaunay_file = os.path.join(output_dir, "advanced_delaunay_mesh.ply")
            o3d.io.write_triangle_mesh(delaunay_file, delaunay_mesh)
            print(f"  Delaunay mesh saved: {delaunay_file}")
        
        # Step 4: Export dense point cloud
        print("\n4. Exporting dense point cloud...")
        pcd_file = os.path.join(output_dir, "advanced_dense_pointcloud.ply")
        pipeline.export_dense_point_cloud(pcd_file)
        
        print(f"\n✓ Advanced reconstruction completed!")
        print(f"  All outputs saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Advanced reconstruction failed: {e}")
        return False


def main():
    """Main demonstration function"""
    print("Dense Reconstruction Methods Demo")
    print("=" * 60)
    
    # Check if images folder is provided
    if len(sys.argv) > 1:
        images_folder = sys.argv[1]
    else:
        # Use default folder if available
        test_folders = ['images', 'test_images', 'data/images']
        images_folder = None
        
        for folder in test_folders:
            if os.path.exists(folder):
                images_folder = folder
                break
        
        if images_folder is None:
            print("Usage: python demo_dense_reconstruction.py <images_folder>")
            print("\nExample: python demo_dense_reconstruction.py /path/to/images")
            print("\nThe images folder should contain a sequence of overlapping images")
            print("taken from different viewpoints of the same object or scene.")
            return
    
    if not os.path.exists(images_folder):
        print(f"Error: Images folder '{images_folder}' does not exist!")
        return
    
    # Count images
    image_files = [f for f in os.listdir(images_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) < 2:
        print(f"Error: Need at least 2 images in '{images_folder}', found {len(image_files)}")
        return
    
    print(f"Found {len(image_files)} images in '{images_folder}'")
    
    # Run demonstrations
    demos = [
        ("PatchMatch Reconstruction", lambda: demo_patchmatch_reconstruction(images_folder)),
        ("SGM Reconstruction", lambda: demo_sgm_reconstruction(images_folder)),
        ("Method Comparison", lambda: demo_comparison(images_folder)),
        ("Advanced Features", lambda: demo_advanced_features(images_folder)),
    ]
    
    for demo_name, demo_func in demos:
        print(f"\n\nRunning {demo_name}...")
        try:
            success = demo_func()
            if success:
                print(f"✓ {demo_name} completed successfully")
            else:
                print(f"✗ {demo_name} failed")
        except Exception as e:
            print(f"✗ {demo_name} crashed: {e}")
        
        print("-" * 60)
    
    print("\nAll demonstrations completed!")
    print("\nGenerated outputs:")
    output_dirs = [
        'output_patchmatch',
        'output_sgm', 
        'output_comparison_patchmatch_tsdf',
        'output_comparison_patchmatch_poisson',
        'output_comparison_sgm_tsdf',
        'output_comparison_sgm_delaunay',
        'output_advanced'
    ]
    
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            print(f"  {output_dir}: {len(files)} files")


if __name__ == "__main__":
    main()