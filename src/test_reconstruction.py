import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from photogrammetry import PhotogrammetryPipeline
import unittest

class TestPhotogrammetryPipeline(unittest.TestCase):
    def test_track_reconstruction(self):
        """
        Runs the photogrammetry pipeline on the specified image directory and checks outputs.
        """
        image_dir = "test_images"
        pipeline = PhotogrammetryPipeline(image_dir)
        mesh, dense_points = pipeline.run_complete_pipeline()

        # Check results using assertions
        self.assertTrue(os.path.exists("race_track_mesh.ply"), "Mesh file was not generated")
        self.assertTrue(os.path.exists("race_track_dense.ply"), "Dense point cloud file was not exported")

        # Assert that mesh and dense_points are not None and have expected structure
        self.assertIsNotNone(mesh, "Mesh output is None")
        self.assertIsNotNone(dense_points, "Dense points output is None")
        # Optionally, check for expected attributes or types
        # For example, if mesh is expected to be a trimesh.Trimesh object:
        # import trimesh
        # self.assertIsInstance(mesh, trimesh.Trimesh, "Mesh is not a Trimesh object")

if __name__ == "__main__":
    unittest.main()
