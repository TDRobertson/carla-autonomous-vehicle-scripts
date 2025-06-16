import unittest
import numpy as np
from ..utils.rotations import skew_symmetric, angle_normalize, omega, Quaternion

class TestRotations(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.vector = np.array([1.0, 2.0, 3.0])
        self.angle = np.pi / 4
        
    def test_skew_symmetric(self):
        """Test skew symmetric matrix creation."""
        # Create skew symmetric matrix
        matrix = skew_symmetric(self.vector)
        
        # Check matrix properties
        self.assertEqual(matrix.shape, (3, 3))
        self.assertTrue(np.allclose(matrix, -matrix.T))  # Anti-symmetric
        
        # Check specific values
        expected = np.array([
            [0.0, -3.0, 2.0],
            [3.0, 0.0, -1.0],
            [-2.0, 1.0, 0.0]
        ])
        np.testing.assert_array_almost_equal(matrix, expected)
        
    def test_angle_normalize(self):
        """Test angle normalization."""
        # Test positive angle
        normalized = angle_normalize(self.angle)
        self.assertAlmostEqual(normalized, self.angle)
        
        # Test negative angle
        normalized = angle_normalize(-self.angle)
        self.assertAlmostEqual(normalized, -self.angle)
        
        # Test angle > 2π
        normalized = angle_normalize(self.angle + 2 * np.pi)
        self.assertAlmostEqual(normalized, self.angle)
        
        # Test angle < -2π
        normalized = angle_normalize(-self.angle - 2 * np.pi)
        self.assertAlmostEqual(normalized, -self.angle)
        
    def test_omega(self):
        """Test quaternion update matrix."""
        # Create omega matrix
        matrix = omega(self.vector)
        
        # Check matrix properties
        self.assertEqual(matrix.shape, (4, 4))
        
        # Check specific values
        expected = np.array([
            [0.0, -1.0, -2.0, -3.0],
            [1.0, 0.0, 3.0, -2.0],
            [2.0, -3.0, 0.0, 1.0],
            [3.0, 2.0, -1.0, 0.0]
        ]) * 0.5
        np.testing.assert_array_almost_equal(matrix, expected)
        
class TestQuaternion(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.q1 = Quaternion()  # Identity quaternion
        self.q2 = Quaternion.from_euler(0, 0, np.pi/2)  # 90° rotation around z
        
    def test_initialization(self):
        """Test quaternion initialization."""
        # Test default initialization
        self.assertTrue(np.allclose(self.q1.to_numpy(), np.array([1.0, 0.0, 0.0, 0.0])))
        
        # Test from array
        q = Quaternion.from_array(np.array([0.5, 0.5, 0.5, 0.5]))
        self.assertTrue(np.allclose(q.to_numpy(), np.array([0.5, 0.5, 0.5, 0.5])))
        
        # Test from euler angles
        expected = np.array([0.7071, 0.0, 0.0, 0.7071])  # cos(45°), 0, 0, sin(45°)
        np.testing.assert_array_almost_equal(self.q2.to_numpy(), expected, decimal=4)
        
    def test_multiplication(self):
        """Test quaternion multiplication."""
        # Multiply two quaternions
        q3 = self.q1 * self.q2
        
        # Check result
        expected = self.q2.to_numpy()  # Identity * q2 = q2
        np.testing.assert_array_almost_equal(q3.to_numpy(), expected)
        
        # Test commutativity
        q4 = self.q2 * self.q1
        np.testing.assert_array_almost_equal(q3.to_numpy(), q4.to_numpy())
        
    def test_conversion(self):
        """Test quaternion conversion methods."""
        # Test to numpy array
        array = self.q1.to_numpy()
        self.assertEqual(array.shape, (4,))
        self.assertTrue(np.allclose(array, np.array([1.0, 0.0, 0.0, 0.0])))
        
        # Test from numpy array
        q = Quaternion.from_array(array)
        self.assertTrue(np.allclose(q.to_numpy(), array))
        
    def test_euler_conversion(self):
        """Test quaternion-euler angle conversion."""
        # Test from euler angles
        q = Quaternion.from_euler(np.pi/4, np.pi/4, np.pi/4)
        
        # Test to euler angles
        roll, pitch, yaw = q.to_euler()
        self.assertAlmostEqual(roll, np.pi/4)
        self.assertAlmostEqual(pitch, np.pi/4)
        self.assertAlmostEqual(yaw, np.pi/4)
        
if __name__ == '__main__':
    unittest.main() 