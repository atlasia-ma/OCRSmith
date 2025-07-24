import pytest 
from unittest.mock import Mock, MagicMock, patch, mock_open
import tempfile 
import os 
import json 
 
from ocrsmith.core import OCRSmithEngine 
from ocrsmith.core.text_placement import PlacementResult 
 
class TestOCRSmithEngine: 
    """Test OCRSmithEngine""" 
     
    @pytest.fixture 
    def mock_engine(self, sample_config): 
        """Create an engine with mocked dependencies""" 
        with patch('ocrsmith.core.OCRSmithEngine.PlacementManager'), \
             patch('ocrsmith.core.OCRSmithEngine.TextDataManager'), \
             patch('ocrsmith.core.OCRSmithEngine.AugmentationPipeline'):
             
            engine = OCRSmithEngine(sample_config) 
             
            # Mock managers 
            engine.background_manager = Mock() 
            engine.font_manager = Mock() 
             
            return engine 
     
    def test_engine_initialization(self, sample_config): 
        with patch('ocrsmith.core.OCRSmithEngine.PlacementManager'), \
             patch('ocrsmith.core.OCRSmithEngine.TextDataManager'), \
             patch('ocrsmith.core.OCRSmithEngine.AugmentationPipeline'):
             
            engine = OCRSmithEngine(sample_config) 
             
            assert engine.config == sample_config 
            assert engine.placement_manager is not None 
            assert engine.text_data_manager is not None 
            assert engine.augmentation_pipeline is not None 
     
    def test_setup_augmentations(self, mock_engine): 
        augmentation_config = { 
            'noise': {'enabled': True, 'factor': 0.1, 'probability': 0.5}, 
            'blur': {'enabled': True, 'radius': 1.0, 'probability': 0.3} 
        } 
         
        mock_engine.setup_augmentations(augmentation_config) 
         
        # Should have called add_augmentation twice 
        assert mock_engine.augmentation_pipeline.add_augmentation.call_count == 2 
     
    @patch('ocrsmith.core.OCRSmithEngine.TextRenderer') 
    @patch('PIL.Image.new') 
    def test_generate_sample(self, mock_image_new, mock_text_renderer_class, mock_engine): 
        # Setup mocks 
        mock_font = Mock() 
        mock_engine.font_manager.load_font.return_value = mock_font 
         
        mock_text_renderer = Mock() 
        mock_text_renderer_class.return_value = mock_text_renderer 
         
        mock_text_image = Mock() 
        mock_text_image.size = (100, 50) 
        mock_text_renderer.generate_text_image.return_value = (mock_text_image, None, (100, 50)) 
         
        mock_background_creator = Mock() 
        mock_background_image = Mock() 
        mock_background_image.convert.return_value = mock_background_image 
        mock_background_creator.render.return_value = mock_background_image 
        mock_engine.background_manager.get_random_background.return_value = mock_background_creator 
         
        mock_engine.text_data_manager.get_random_text.return_value = "Test text" 
         
        # Mock placement result 
        mock_composed_image = Mock() 
        mock_placement_result = PlacementResult( 
            mock_composed_image,  
            (10, 20, 110, 70),  
            {'placement_type': 'test'} 
        ) 
        mock_engine.placement_manager.place_text.return_value = mock_placement_result 
         
        mock_engine.augmentation_pipeline.apply_all.return_value = mock_composed_image 
         
        # Test generate_sample 
        result = mock_engine.generate_sample(placement_strategy='random') 
         
        image, text, bbox, metadata = result 
         
        assert image == mock_composed_image 
        assert text == "Test text" 
        assert bbox == (10, 20, 110, 70) 
        assert metadata == {'placement_type': 'test'} 
     
    def test_generate_sample_no_text_raises_error(self, mock_engine): 
        mock_engine.text_data_manager.get_random_text.return_value = None 
         
        with pytest.raises(ValueError, match="No text provided"): 
            mock_engine.generate_sample() 
     
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
     
    @patch('os.makedirs') 
    @patch('builtins.open', new_callable=mock_open) 
    @patch('json.dump') 
    def test_generate_dataset(self, mock_json_dump, mock_file, mock_makedirs, mock_engine, temp_dir): 
        # Mock generate_sample to return consistent results 
        mock_image = Mock() 
        mock_image.save = Mock() 
         
        def mock_generate_sample(**kwargs): 
            return mock_image, "test text", (0, 0, 100, 50), {'type': 'test'} 
         
        mock_engine.generate_sample = Mock(side_effect=mock_generate_sample) 
         
        # Test generate_dataset 
        annotations = mock_engine.generate_dataset(num_samples=2, output_dir=temp_dir) 
         
        # Verify results 
        assert len(annotations) == 2 
        assert mock_engine.generate_sample.call_count == 2 
        assert mock_image.save.call_count == 2 
        mock_makedirs.assert_called_once_with(temp_dir, exist_ok=True) 
        mock_json_dump.assert_called_once() 
        
        # Verify the structure of annotations
        for annotation in annotations:
            assert 'image_path' in annotation
            assert 'text' in annotation
            assert 'bbox' in annotation
            assert 'placement_metadata' in annotation
     
    def test_generate_dataset_handles_errors(self, mock_engine, temp_dir): 
        # Mock generate_sample to raise an exception 
        mock_engine.generate_sample = Mock(side_effect=Exception("Test error")) 
         
        with patch('builtins.print') as mock_print: 
            annotations = mock_engine.generate_dataset(num_samples=2, output_dir=temp_dir) 
         
        # Should continue despite errors 
        assert len(annotations) == 0 
        assert mock_print.call_count >= 2  # Error messages
        
    def test_generate_dataset_creates_valid_filenames(self, mock_engine, temp_dir):
        """Test that generated filenames follow expected pattern"""
        mock_image = Mock()
        mock_image.save = Mock()
        
        def mock_generate_sample(**kwargs):
            return mock_image, "test text", (0, 0, 100, 50), {'type': 'test'}
        
        mock_engine.generate_sample = Mock(side_effect=mock_generate_sample)
        
        with patch('os.makedirs'), \
             patch('builtins.open', mock_open()), \
             patch('json.dump'):
            
            annotations = mock_engine.generate_dataset(num_samples=3, output_dir=temp_dir)
        
        # Check that image paths are properly formatted
        expected_paths = [
            os.path.join(temp_dir, 'sample_000000.png'),
            os.path.join(temp_dir, 'sample_000001.png'), 
            os.path.join(temp_dir, 'sample_000002.png')
        ]
        actual_paths = [ann['image_path'] for ann in annotations]
        
        assert actual_paths == expected_paths