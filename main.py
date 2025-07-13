
#!/usr/bin/env python3
"""
Main entry point for the Enhanced Photogrammetry Pipeline
Provides guidance for development and testing
"""

import os
import sys

def main():
    """Main function to guide users on how to use the pipeline"""
    print("=" * 60)
    print("Enhanced Photogrammetry Pipeline - Development Environment")
    print("=" * 60)
    
    # Check for images directory
    if os.path.exists('images') and os.listdir('images'):
        image_files = [f for f in os.listdir('images') 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(image_files) >= 2:
            print(f"✓ Found {len(image_files)} images in 'images/' directory")
            print("\nRunning dense reconstruction demo...")
            
            # Add src to path
            sys.path.append('src')
            
            # Import and run demo
            try:
                from demo_dense_reconstruction import main as demo_main
                sys.argv = ['demo_dense_reconstruction.py', 'images']
                demo_main()
            except Exception as e:
                print(f"Error running demo: {e}")
                print("\nYou can also run the demo manually:")
                print("python3 src/demo_dense_reconstruction.py images")
        else:
            print(f"✗ Found only {len(image_files)} images in 'images/' directory")
            print("  Need at least 2 images for reconstruction")
    else:
        print("✗ No 'images' directory found or it's empty")
    
    print("\n" + "=" * 60)
    print("USAGE INSTRUCTIONS")
    print("=" * 60)
    print("1. Add 2+ overlapping images to the 'images/' directory")
    print("2. Click the Run button to start the reconstruction demo")
    print("\nManual Usage:")
    print("• python3 src/demo_dense_reconstruction.py /path/to/images")
    print("• python3 src/photogrammetry_v2.py /path/to/images --dense")
    print("\nProgrammatic Usage:")
    print("from src.enhanced_photogrammetry import EnhancedPhotogrammetryPipeline")
    print("pipeline = EnhancedPhotogrammetryPipeline('images/')")
    print("mesh = pipeline.run_complete_dense_pipeline()")
    
    print("\n" + "=" * 60)
    print("AVAILABLE RECONSTRUCTION METHODS")
    print("=" * 60)
    print("• PatchMatch MVS - High quality depth estimation")
    print("• Semi-Global Matching - Fast stereo processing") 
    print("• TSDF Fusion - Watertight mesh generation")
    print("• Delaunay Triangulation - Fast mesh creation")
    print("• Poisson Reconstruction - Smooth surfaces")
    
    if not os.path.exists('images'):
        print("\n" + "=" * 60)
        print("QUICK START")
        print("=" * 60)
        print("Create an 'images' directory and add your image sequence:")
        print("mkdir images")
        print("# Copy your images to the images/ directory")
        print("# Then click Run to start reconstruction")

if __name__ == "__main__":
    main()
