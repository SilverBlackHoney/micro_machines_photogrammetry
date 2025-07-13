
# Enhanced Photogrammetry Pipeline - Demo Instructions

Welcome to the Enhanced Photogrammetry Pipeline! This guide will walk you through the complete process of creating 3D reconstructions from your photos.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Adding Your Photos](#adding-your-photos)
3. [Running the Demo](#running-the-demo)
4. [Understanding the Outputs](#understanding-the-outputs)
5. [Advanced Usage](#advanced-usage)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Step 1: Add Your Photos
1. Create an `images` folder in the project root (if it doesn't exist)
2. Add at least 2-3 overlapping photos of your object/scene
3. Click the **Run** button in Replit

That's it! The demo will automatically process your images and generate 3D reconstructions.

## Adding Your Photos

### Photo Requirements
- **Minimum**: 2 photos (recommended: 5-20 photos)
- **Format**: JPG, JPEG, or PNG
- **Overlap**: Each photo should overlap with at least one other photo by 30-70%
- **Quality**: Good lighting, sharp focus, minimal motion blur

### Photo Tips for Best Results
✅ **Good practices:**
- Walk around your object taking photos from different angles
- Keep the object in frame across all photos
- Maintain consistent lighting
- Use a tripod or steady hands
- Take photos at multiple heights/elevations

❌ **Avoid:**
- Blurry or dark photos
- Photos with completely different lighting
- Photos too close or too far from the object
- Photos with large gaps in coverage

### Adding Photos to Replit

**Method 1: Drag and Drop**
1. Create an `images` folder by clicking the "Add folder" button
2. Drag your photos directly into the `images` folder in the file tree

**Method 2: Upload Button**
1. Click the three dots (⋯) next to the `images` folder
2. Select "Upload file"
3. Choose your photos

**Method 3: Import from URL**
1. Right-click in the `images` folder
2. Select "Upload file"
3. Use "Upload from URL" if your photos are online

## Running the Demo

### Option 1: Using Replit (Recommended)

1. **Ensure your photos are in the `images` folder**
2. **Click the Run button** - This runs the complete demo automatically
3. **Watch the progress** in the console output
4. **Check the generated files** in the output folders

### Option 2: Command Line Interface

Open the Shell tab in Replit and run:

```bash
# Run complete demo with all methods
python3 src/demo_dense_reconstruction.py images

# Run basic SfM only (faster)
python3 src/photogrammetry.py images

# Run with specific options
python3 src/photogrammetry.py images --dense --feature_type SIFT
```

### Available Command Line Options

```bash
# Basic reconstruction
python3 src/photogrammetry.py images

# Dense reconstruction
python3 src/photogrammetry.py images --dense

# Use SIFT features (higher quality)
python3 src/photogrammetry.py images --feature_type SIFT

# Use CUDA acceleration (if available)
python3 src/photogrammetry.py images --use_cuda

# Custom focal length
python3 src/photogrammetry.py images --focal_length 800
```

## Understanding the Outputs

### Generated Files and Folders

After running the demo, you'll find these outputs:

#### Main Output Files
- **`reconstruction.ply`** - Sparse 3D point cloud from SfM
- **`dense_reconstruction.ply`** - Dense 3D point cloud
- **`reconstruction_mesh.ply`** - Generated 3D mesh

#### Demo Output Folders
- **`output_patchmatch/`** - High-quality PatchMatch results
- **`output_sgm/`** - Fast Semi-Global Matching results
- **`output_comparison_*/`** - Method comparison results
- **`output_advanced/`** - Advanced reconstruction examples

#### Individual Output Folder Contents
Each output folder contains:
- **`detailed_mesh_*.ply`** - Main reconstructed mesh
- **`dense_point_cloud.ply`** - Dense point cloud
- **`reconstruction_summary.png`** - Visualization summary
- **Depth maps** (optional) - Individual camera depth maps

### Viewing Your Results

#### Option 1: Download and View Locally
1. Right-click on any `.ply` file
2. Select "Download"
3. Open with software like:
   - [MeshLab](https://www.meshlab.net/) (free)
   - [CloudCompare](https://www.cloudcompare.org/) (free)
   - [Blender](https://www.blender.org/) (free)

#### Option 2: Online Viewers
1. Download the `.ply` file
2. Upload to online viewers like:
   - [3D Viewer Online](https://3dviewer.net/)
   - [ViewSTL](https://www.viewstl.com/)

#### Option 3: Code Visualization
Add this to the end of your script to view in Replit:
```python
import open3d as o3d

# Load and display point cloud
pcd = o3d.io.read_point_cloud("reconstruction.ply")
o3d.visualization.draw_geometries([pcd])

# Load and display mesh
mesh = o3d.io.read_triangle_mesh("reconstruction_mesh.ply")
o3d.visualization.draw_geometries([mesh])
```

## Advanced Usage

### Programmatic Usage

```python
from src.enhanced_photogrammetry import EnhancedPhotogrammetryPipeline

# Initialize pipeline
pipeline = EnhancedPhotogrammetryPipeline("images/")

# Run complete dense reconstruction
mesh = pipeline.run_complete_dense_pipeline(
    reconstruction_method='patchmatch',  # or 'sgm'
    meshing_method='tsdf',               # or 'poisson', 'delaunay'
    output_dir='my_output'
)

# Run specific methods
pipeline.run_dense_reconstruction(
    method='patchmatch',
    patch_size=9,
    iterations=5
)

# Generate different mesh types
tsdf_mesh = pipeline.generate_detailed_mesh(method='tsdf')
poisson_mesh = pipeline.generate_detailed_mesh(method='poisson')
```

### Method Comparison

| Method | Speed | Quality | Memory | Best For |
|--------|-------|---------|--------|----------|
| **PatchMatch MVS** | Medium | High | Medium | High-quality models |
| **Semi-Global Matching** | Fast | Medium | Low | Quick previews |
| **TSDF Fusion** | Medium | High | High | Watertight meshes |
| **Poisson Reconstruction** | Fast | High | Medium | Smooth surfaces |
| **Delaunay Triangulation** | Fast | Medium | Low | Point cloud meshing |

### Parameter Tuning

For better results, try adjusting these parameters:

```python
# High quality settings (slower)
mesh = pipeline.run_complete_dense_pipeline(
    reconstruction_method='patchmatch',
    meshing_method='tsdf',
    patch_size=9,           # Larger patches
    iterations=5,           # More iterations
    voxel_size=0.005,      # Finer voxels
    post_process=True       # Enable cleaning
)

# Fast settings (lower quality)
mesh = pipeline.run_complete_dense_pipeline(
    reconstruction_method='sgm',
    meshing_method='delaunay',
    max_disparity=64,       # Smaller search range
    post_process=False      # Skip cleaning
)
```

## Troubleshooting

### Common Issues

#### "No valid image pairs found"
- **Cause**: Photos don't overlap enough
- **Solution**: Take more photos with better overlap (30-70%)

#### "Empty point cloud generated"
- **Cause**: Poor feature matching or camera calibration
- **Solution**: 
  - Use better lit, sharper photos
  - Try SIFT features: `--feature_type SIFT`
  - Adjust focal length: `--focal_length 1000`

#### "Mesh generation failed"
- **Cause**: Insufficient dense points
- **Solution**:
  - Use more photos
  - Try different meshing method
  - Adjust depth range parameters

#### "Out of memory error"
- **Cause**: Images too large or too many
- **Solution**:
  - Resize images to 1920x1080 or smaller
  - Use fewer images (10-15 max)
  - Try SGM instead of PatchMatch

### Performance Tips

#### For Speed:
- Use SGM reconstruction: `reconstruction_method='sgm'`
- Use Delaunay meshing: `meshing_method='delaunay'`
- Resize images to 1280x720
- Use fewer images (5-10)

#### For Quality:
- Use PatchMatch reconstruction: `reconstruction_method='patchmatch'`
- Use TSDF meshing: `meshing_method='tsdf'`
- Use SIFT features: `--feature_type SIFT`
- Use more photos (10-20)
- Use higher resolution images

#### For Memory Efficiency:
- Process in batches
- Use coarser voxel grids
- Limit depth range
- Enable point cloud downsampling

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Then run your pipeline
pipeline = EnhancedPhotogrammetryPipeline("images/")
mesh = pipeline.run_complete_dense_pipeline()
```

### Getting Help

If you encounter issues:

1. **Check the console output** for error messages
2. **Verify your photos** meet the requirements
3. **Try different parameters** or methods
4. **Use debug mode** for detailed information
5. **Start with fewer, simpler photos** to test

### Example Workflows

#### Quick Test (2-3 minutes)
```bash
python3 src/photogrammetry.py images --feature_type ORB
```

#### High Quality (10-20 minutes)
```bash
python3 src/demo_dense_reconstruction.py images
```

#### Custom Processing
```python
from src.enhanced_photogrammetry import EnhancedPhotogrammetryPipeline

pipeline = EnhancedPhotogrammetryPipeline("images/")
mesh = pipeline.run_complete_dense_pipeline(
    reconstruction_method='patchmatch',
    meshing_method='tsdf',
    output_dir='custom_output'
)
```

## Sample Results

With good input photos, you can expect:
- **Sparse point cloud**: 1,000-10,000 points
- **Dense point cloud**: 50,000-500,000 points  
- **Mesh**: 10,000-100,000 triangles
- **Processing time**: 2-30 minutes depending on method and image count

The pipeline will automatically generate visualization summaries showing your reconstruction progress and results.

---

**Happy reconstructing! 🚀**

For more technical details, see the [DENSE_RECONSTRUCTION_README.md](DENSE_RECONSTRUCTION_README.md) file.
