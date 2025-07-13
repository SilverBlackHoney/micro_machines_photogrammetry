
# Enhanced Photogrammetry Pipeline

A powerful Python-based photogrammetry pipeline for creating high-quality 3D reconstructions from image sequences. This project implements state-of-the-art computer vision algorithms including Structure from Motion (SfM), Multi-View Stereo (MVS), and advanced mesh generation techniques.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)](https://opencv.org/)
[![Open3D](https://img.shields.io/badge/Open3D-0.13+-orange.svg)](http://www.open3d.org/)

## 🚀 Features

### Core Photogrammetry
- **Structure from Motion (SfM)** - Automatic camera pose estimation and sparse 3D reconstruction
- **Feature Detection** - SIFT and ORB feature extractors with CUDA acceleration support
- **Bundle Adjustment** - Iterative refinement of camera poses and 3D points
- **Camera Calibration** - Automatic intrinsic parameter estimation

### Advanced Dense Reconstruction
- **PatchMatch Multi-View Stereo** - High-quality dense depth estimation
- **Semi-Global Matching (SGM)** - Fast stereo processing for real-time applications
- **TSDF Volume Fusion** - Watertight mesh generation using Truncated Signed Distance Fields
- **Multi-Resolution Fusion** - Intelligent point cloud fusion with confidence weighting

### Mesh Generation
- **Delaunay Triangulation** - Fast mesh creation from point clouds
- **Poisson Surface Reconstruction** - Smooth, high-quality surface generation
- **Advanced Post-Processing** - Mesh cleaning, smoothing, and optimization
- **Multiple Output Formats** - PLY, OBJ, and other standard 3D formats

### Quality Features
- **Confidence Maps** - Per-pixel reliability estimation
- **Outlier Detection** - Automatic filtering of erroneous reconstructions
- **Visualization Tools** - Comprehensive reconstruction summaries and plots
- **Modular Design** - Mix and match different algorithms for optimal results

## 📋 Requirements

- Python 3.8 or higher
- OpenCV 4.5+ (with contrib modules)
- NumPy 1.20+
- SciPy 1.7+
- Open3D 0.13+
- Matplotlib 3.5+

## 🛠️ Installation

### Quick Start on Replit
1. **Fork this Repl** or create a new Python Repl
2. **Upload your images** to the `images/` folder
3. **Click the Run button** to start the demo

### Local Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/enhanced-photogrammetry.git
cd enhanced-photogrammetry

# Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Start

### 1. Prepare Your Images
- Add 2+ overlapping images to the `images/` folder
- Ensure good overlap (30-70%) between consecutive images
- Use consistent lighting and sharp focus

### 2. Run the Demo
```bash
# Complete dense reconstruction demo
python3 src/demo_dense_reconstruction.py images

# Basic SfM reconstruction only
python3 src/photogrammetry.py images

# Custom parameters
python3 src/photogrammetry.py images --feature_type SIFT --dense
```

### 3. View Results
Generated files will be saved in output folders:
- `*.ply` files - 3D point clouds and meshes
- `reconstruction_summary.png` - Visualization summary
- Multiple output folders for different methods

## 📚 Usage Examples

### Basic Reconstruction
```python
from src.photogrammetry import PhotogrammetryPipeline

# Initialize pipeline
pipeline = PhotogrammetryPipeline("images/")

# Run Structure from Motion
n_poses, n_points = pipeline.run_sfm_pipeline(feature_type='SIFT')
print(f"Reconstructed {n_poses} camera poses and {n_points} 3D points")
```

### Enhanced Dense Reconstruction
```python
from src.enhanced_photogrammetry import EnhancedPhotogrammetryPipeline

# Initialize enhanced pipeline
pipeline = EnhancedPhotogrammetryPipeline("images/")

# Run complete dense reconstruction
mesh = pipeline.run_complete_dense_pipeline(
    reconstruction_method='patchmatch',  # High quality
    meshing_method='tsdf',               # Watertight mesh
    output_dir='my_reconstruction'
)

# Generate high-detail mesh with custom parameters
detailed_mesh = pipeline.generate_detailed_mesh(
    method='poisson',
    depth=10,                    # High detail level
    post_process=True           # Apply mesh cleaning
)
```

### Method Comparison
```python
# Compare different reconstruction methods
methods = [
    ('patchmatch', 'tsdf'),     # High quality
    ('sgm', 'delaunay'),        # Fast processing
    ('patchmatch', 'poisson')   # Smooth surfaces
]

for depth_method, mesh_method in methods:
    mesh = pipeline.run_complete_dense_pipeline(
        reconstruction_method=depth_method,
        meshing_method=mesh_method,
        output_dir=f'output_{depth_method}_{mesh_method}'
    )
```

## 🔧 Command Line Interface

### Basic Usage
```bash
# Run with default settings (ORB features, basic reconstruction)
python3 src/photogrammetry.py images/

# Use SIFT features for higher quality
python3 src/photogrammetry.py images/ --feature_type SIFT

# Enable dense reconstruction
python3 src/photogrammetry.py images/ --dense

# Use CUDA acceleration (if available)
python3 src/photogrammetry.py images/ --use_cuda

# Custom focal length
python3 src/photogrammetry.py images/ --focal_length 800
```

### Advanced Options
```bash
# Complete demo with all methods
python3 src/demo_dense_reconstruction.py images/

# Run specific reconstruction method
python3 -c "
from src.enhanced_photogrammetry import EnhancedPhotogrammetryPipeline
pipeline = EnhancedPhotogrammetryPipeline('images/')
mesh = pipeline.run_complete_dense_pipeline(
    reconstruction_method='patchmatch',
    meshing_method='tsdf'
)
"
```

## 📊 Reconstruction Methods

| Method | Speed | Quality | Memory | Best For |
|--------|-------|---------|--------|----------|
| **PatchMatch MVS** | Medium | High | Medium | High-quality models |
| **Semi-Global Matching** | Fast | Medium | Low | Quick previews |
| **TSDF Fusion** | Medium | High | High | Watertight meshes |
| **Poisson Reconstruction** | Fast | High | Medium | Smooth surfaces |
| **Delaunay Triangulation** | Fast | Medium | Low | Point cloud meshing |

## 🎨 Output Examples

The pipeline generates various outputs:

### Sparse Reconstruction (SfM)
- **Point Cloud**: 1,000-10,000 sparse feature points
- **Camera Poses**: Estimated positions and orientations
- **Reprojection Errors**: Quality metrics

### Dense Reconstruction
- **Dense Point Cloud**: 50,000-500,000+ points
- **Depth Maps**: Per-view depth estimation
- **Confidence Maps**: Reliability information

### Mesh Generation
- **Triangle Meshes**: 10,000-100,000+ triangles
- **Textured Models**: Color information preserved
- **Watertight Surfaces**: Ready for 3D printing

## 🔍 Technical Details

### Algorithms Implemented

#### Structure from Motion
- Feature detection and matching (SIFT, ORB)
- Essential matrix estimation with RANSAC
- Camera pose recovery and triangulation
- Bundle adjustment optimization

#### Dense Reconstruction
- **PatchMatch MVS**: Iterative depth refinement with spatial propagation
- **Semi-Global Matching**: Dynamic programming stereo matching
- **TSDF Fusion**: Volumetric surface integration
- **Multi-Resolution Fusion**: Confidence-weighted point cloud merging

#### Mesh Processing
- Delaunay triangulation with visibility constraints
- Poisson surface reconstruction with normal estimation
- Advanced post-processing (smoothing, hole filling, decimation)
- Mesh quality assessment and repair

### Performance Optimization
- **CUDA Support**: GPU acceleration for feature detection
- **Multi-threading**: Parallel processing where applicable
- **Memory Management**: Efficient handling of large datasets
- **Adaptive Parameters**: Automatic parameter tuning based on input

## 📁 Project Structure

```
enhanced-photogrammetry/
├── src/
│   ├── photogrammetry.py           # Main SfM pipeline
│   ├── enhanced_photogrammetry.py  # Enhanced pipeline with dense reconstruction
│   ├── dense_reconstruction.py     # Dense reconstruction algorithms
│   ├── demo_dense_reconstruction.py # Comprehensive demo script
│   └── test_reconstruction.py      # Unit tests
├── images/                         # Input images directory
├── requirements.txt                # Python dependencies
├── DEMO_INSTRUCTIONS.md           # Detailed usage guide
├── DENSE_RECONSTRUCTION_README.md # Technical documentation
└── README.md                      # This file
```

## 🚨 Troubleshooting

### Common Issues

#### "No valid image pairs found"
- **Cause**: Insufficient image overlap
- **Solution**: Take more photos with 30-70% overlap between consecutive images

#### "Empty point cloud generated"
- **Cause**: Poor feature matching
- **Solution**: Use SIFT features (`--feature_type SIFT`) or adjust focal length

#### "Out of memory error"
- **Cause**: Images too large or too many
- **Solution**: Resize images to 1920x1080 or use fewer images (10-15 max)

#### "Mesh generation failed"
- **Cause**: Insufficient dense points
- **Solution**: Use more images or try different meshing methods

### Performance Tips

#### For Speed:
- Use SGM reconstruction with Delaunay meshing
- Resize images to 1280x720 or smaller
- Use ORB features instead of SIFT
- Process fewer images (5-10)

#### For Quality:
- Use PatchMatch reconstruction with TSDF meshing
- Use SIFT features for better matching
- Use higher resolution images
- Take more photos (15-20) with good overlap

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests if applicable
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Areas for Contribution
- Neural network-based depth estimation
- Real-time processing optimizations
- Additional mesh processing algorithms
- Improved GUI/visualization tools
- Documentation and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenCV community for computer vision algorithms
- Open3D team for 3D processing capabilities
- Research papers and implementations that inspired this work
- Contributors and users who provide feedback and improvements

## 📞 Support

- **Documentation**: See [DEMO_INSTRUCTIONS.md](DEMO_INSTRUCTIONS.md) for detailed usage
- **Technical Details**: Check [DENSE_RECONSTRUCTION_README.md](DENSE_RECONSTRUCTION_README.md)
- **Issues**: Report bugs and feature requests on GitHub Issues
- **Discussions**: Join project discussions for questions and ideas

## 🔮 Roadmap

### Upcoming Features
- [ ] Neural depth estimation integration
- [ ] Real-time processing pipeline
- [ ] Web-based visualization interface
- [ ] Multi-scale reconstruction
- [ ] Texture mapping improvements
- [ ] Mobile device support

### Long-term Goals
- [ ] Integration with popular 3D software
- [ ] Cloud-based processing options
- [ ] AR/VR visualization support
- [ ] Professional workflow tools
- [ ] Educational materials and tutorials

---

**Star this repository if you find it useful!** ⭐

For questions, suggestions, or collaboration opportunities, feel free to open an issue or reach out.
