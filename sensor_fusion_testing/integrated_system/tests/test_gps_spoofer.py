import unittest
import numpy as np
from ..core.gps_spoofer import GPSSpoofer, SpoofingStrategy

class TestGPSSpoofer(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.spoofer = GPSSpoofer()
        self.true_position = np.array([0.0, 0.0, 0.0])
        
    def test_no_spoofing(self):
        """Test with no spoofing strategy."""
        # Set no spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.NONE)
        
        # Get spoofed position
        spoofed_position = self.spoofer.spoof_position(self.true_position)
        
        # Check that position is unchanged
        np.testing.assert_array_almost_equal(
            spoofed_position,
            self.true_position
        )
        
        # Check spoofing status
        active, strategy = self.spoofer.get_spoofing_status()
        self.assertFalse(active)
        self.assertEqual(strategy, SpoofingStrategy.NONE.value)
        
    def test_jump_spoofing(self):
        """Test jump spoofing strategy."""
        # Set jump spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.JUMP)
        
        # Get spoofed position
        spoofed_position = self.spoofer.spoof_position(self.true_position)
        
        # Check that position has changed
        self.assertFalse(np.array_equal(spoofed_position, self.true_position))
        
        # Check that change is significant
        position_change = np.linalg.norm(spoofed_position - self.true_position)
        self.assertGreater(position_change, self.spoofer.jump_magnitude * 0.5)
        
        # Check spoofing status
        active, strategy = self.spoofer.get_spoofing_status()
        self.assertTrue(active)
        self.assertEqual(strategy, SpoofingStrategy.JUMP.value)
        
    def test_drift_spoofing(self):
        """Test drift spoofing strategy."""
        # Set drift spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.DRIFT)
        
        # Get spoofed position
        spoofed_position = self.spoofer.spoof_position(self.true_position)
        
        # Check that position has changed
        self.assertFalse(np.array_equal(spoofed_position, self.true_position))
        
        # Check that change is in drift direction
        position_change = spoofed_position - self.true_position
        direction = position_change / np.linalg.norm(position_change)
        expected_direction = self.spoofer.sequential_direction
        np.testing.assert_array_almost_equal(direction, expected_direction)
        
        # Check spoofing status
        active, strategy = self.spoofer.get_spoofing_status()
        self.assertTrue(active)
        self.assertEqual(strategy, SpoofingStrategy.DRIFT.value)
        
    def test_random_spoofing(self):
        """Test random spoofing strategy."""
        # Set random spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.RANDOM)
        
        # Get spoofed position
        spoofed_position = self.spoofer.spoof_position(self.true_position)
        
        # Check that position has changed
        self.assertFalse(np.array_equal(spoofed_position, self.true_position))
        
        # Check that change is within expected range
        position_change = np.linalg.norm(spoofed_position - self.true_position)
        self.assertLess(position_change, self.spoofer.random_std * 3)
        
        # Check spoofing status
        active, strategy = self.spoofer.get_spoofing_status()
        self.assertTrue(active)
        self.assertEqual(strategy, SpoofingStrategy.RANDOM.value)
        
    def test_sequential_spoofing(self):
        """Test sequential spoofing strategy."""
        # Set sequential spoofing strategy
        self.spoofer.set_strategy(SpoofingStrategy.SEQUENTIAL)
        
        # Get first spoofed position
        spoofed_position1 = self.spoofer.spoof_position(self.true_position)
        
        # Get second spoofed position
        spoofed_position2 = self.spoofer.spoof_position(self.true_position)
        
        # Check that positions are different
        self.assertFalse(np.array_equal(spoofed_position1, spoofed_position2))
        
        # Check that change is in sequential direction
        position_change = spoofed_position2 - spoofed_position1
        direction = position_change / np.linalg.norm(position_change)
        expected_direction = self.spoofer.sequential_direction
        np.testing.assert_array_almost_equal(direction, expected_direction)
        
        # Check that step size is correct
        step_size = np.linalg.norm(position_change)
        self.assertAlmostEqual(step_size, self.spoofer.sequential_step)
        
        # Check spoofing status
        active, strategy = self.spoofer.get_spoofing_status()
        self.assertTrue(active)
        self.assertEqual(strategy, SpoofingStrategy.SEQUENTIAL.value)
        
if __name__ == '__main__':
    unittest.main() 