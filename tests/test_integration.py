import os

import pandas as pd


class TestIntegration:
    """Integration tests for OCRSmith components"""

    def test_end_to_end_pipeline(self, temp_dir):
        """Test complete pipeline from text loading to image generation"""
        # This would be a more complex test that uses real components
        # For now, we'll test that the structure supports integration

        # Create test data
        test_texts = ["Hello world", "Test text", "مرحبا"]
        csv_data = pd.DataFrame({"text": test_texts})
        csv_path = os.path.join(temp_dir, "test.csv")
        csv_data.to_csv(csv_path, index=False)

        # Test that file was created properly
        assert os.path.exists(csv_path)

        # Load and verify
        loaded_data = pd.read_csv(csv_path)
        assert len(loaded_data) == 3
        assert loaded_data["text"].tolist() == test_texts
