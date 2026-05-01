"""
Phase 1 Validation Tests
=========================

Validates that all Phase 1 components are working correctly.
Tests dummy data generation, dataset loading, and shape validation.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.dummy_dataset import generate_dummy_batch, validate_batch_shapes
from data.dataset import BasicDataset, create_dummy_dataloader


def print_header(text: str, length: int = 80) -> None:
    """Print a formatted header."""
    print("\n" + "=" * length)
    print(text.center(length))
    print("=" * length)


def print_section(text: str) -> None:
    """Print a section header."""
    print(f"\n[{text}]")


def test_dummy_data_generation():
    """Test 1: Dummy data generation"""
    print_section("TEST 1: Dummy Data Generation")

    try:
        batch = generate_dummy_batch(batch_size=4, seed=42)
        
        # Check keys
        assert 'frames' in batch, "Missing 'frames' key"
        assert 'mel' in batch, "Missing 'mel' key"
        assert 'labels' in batch, "Missing 'labels' key"
        
        # Check shapes — visual: (B, T, 6, 224, 224), audio: (B, T, 80, F)
        assert batch['frames'].shape == (4, 16, 6, 224, 224), \
            f"Wrong frames shape: {batch['frames'].shape}"
        assert batch['mel'].shape == (4, 16, 80, 32), \
            f"Wrong mel shape: {batch['mel'].shape}"
        assert batch['labels'].shape == (4,), \
            f"Wrong labels shape: {batch['labels'].shape}"
        
        # Check data types
        assert batch['frames'].dtype == torch.float32, "Wrong frames dtype"
        assert batch['mel'].dtype == torch.float32, "Wrong mel dtype"
        assert batch['labels'].dtype == torch.long, "Wrong labels dtype"
        
        # Check value ranges
        assert batch['frames'].min() >= 0 and batch['frames'].max() <= 1, \
            "Visual data out of range [0, 1]"
        assert batch['mel'].min() >= 0 and batch['mel'].max() <= 1, \
            "Audio data out of range [0, 1]"
        
        print("    ✅ Dummy data generation PASSED")
        return True
    
    except Exception as e:
        print(f"    ❌ Dummy data generation FAILED: {e}")
        return False


def test_batch_shape_validation():
    """Test 2: Batch shape validation"""
    print_section("TEST 2: Batch Shape Validation")

    try:
        batch = generate_dummy_batch(batch_size=32)
        
        is_valid = validate_batch_shapes(
            batch,
            expected_frames_shape=(32, 16, 6, 224, 224),
            expected_mel_shape=(32, 16, 80, 32),
            expected_labels_shape=(32,)
        )
        
        assert is_valid, "Shape validation failed"
        print("    ✅ Batch shape validation PASSED")
        return True
    
    except Exception as e:
        print(f"    ❌ Batch shape validation FAILED: {e}")
        return False


def test_balanced_labels():
    """Test 3: Balanced label distribution"""
    print_section("TEST 3: Balanced Label Distribution")

    try:
        batch = generate_dummy_batch(batch_size=100, balanced=True)
        unique, counts = torch.unique(batch['labels'], return_counts=True)
        
        # Check for both classes
        assert len(unique) == 2, "Should have 2 classes (real/fake)"
        
        # Check balance (allowing ±2 sample tolerance for odd batch sizes)
        label_0_count = (batch['labels'] == 0).sum().item()
        label_1_count = (batch['labels'] == 1).sum().item()
        diff = abs(label_0_count - label_1_count)
        
        assert diff <= 2, f"Label imbalance too large: {label_0_count} vs {label_1_count}"
        
        print(f"    Label 0 (Real): {label_0_count} samples")
        print(f"    Label 1 (Fake): {label_1_count} samples")
        print("    ✅ Balanced label distribution PASSED")
        return True
    
    except Exception as e:
        print(f"    ❌ Balanced label distribution FAILED: {e}")
        return False


def test_dataset_class():
    """Test 4: BasicDataset class"""
    print_section("TEST 4: BasicDataset Class")

    try:
        dataset = BasicDataset(num_samples=50, use_dummy_data=True)
        
        assert len(dataset) == 50, f"Wrong dataset size: {len(dataset)}"
        
        sample = dataset[0]
        assert 'frames' in sample, "Missing 'frames' in sample"
        assert 'mel' in sample, "Missing 'mel' in sample"
        assert 'label' in sample, "Missing 'label' in sample"
        
        # Single sample shapes (no batch dimension)
        assert sample['frames'].shape == (16, 6, 224, 224), \
            f"Wrong sample frames shape: {sample['frames'].shape}"
        assert sample['mel'].shape == (16, 80, 32), \
            f"Wrong sample mel shape: {sample['mel'].shape}"
        assert sample['label'].shape == (1,), \
            f"Wrong sample label shape: {sample['label'].shape}"
        
        print(f"    Dataset length: {len(dataset)}")
        print(f"    Sample frames shape: {sample['frames'].shape}")
        print(f"    Sample mel shape:    {sample['mel'].shape}")
        print(f"    Sample label shape:  {sample['label'].shape}")
        print("    ✅ BasicDataset class PASSED")
        return True
    
    except Exception as e:
        print(f"    ❌ BasicDataset class FAILED: {e}")
        return False


def test_dataloader():
    """Test 5: DataLoader creation and iteration"""
    print_section("TEST 5: DataLoader Creation and Iteration")

    try:
        dataloader = create_dummy_dataloader(
            num_samples=64,
            batch_size=32,
            balanced_sampling=True
        )
        
        assert len(dataloader) > 0, "DataLoader is empty"
        
        # Iterate through first batch
        batch = next(iter(dataloader))
        
        assert batch['frames'].shape[0] <= 32, "Batch size exceeds maximum"
        assert batch['mel'].shape[0] == batch['frames'].shape[0], \
            "Audio and visual batch sizes don't match"
        
        # Check batch shapes include the F (time frames) dimension for mel
        assert len(batch['mel'].shape) == 4, \
            f"Mel should be 4D (B, T, 80, F), got {batch['mel'].shape}"
        
        print(f"    DataLoader batches: {len(dataloader)}")
        print(f"    First batch frames shape: {batch['frames'].shape}")
        print(f"    First batch mel shape:    {batch['mel'].shape}")
        print(f"    First batch label shape:  {batch['label'].shape}")
        print("    ✅ DataLoader PASSED")
        return True
    
    except Exception as e:
        print(f"    ❌ DataLoader FAILED: {e}")
        return False


def test_edge_cases():
    """Test 6: Edge cases and error handling"""
    print_section("TEST 6: Edge Cases and Error Handling")

    try:
        # Test with batch_size=1
        batch = generate_dummy_batch(batch_size=1)
        assert batch['frames'].shape == (1, 16, 6, 224, 224), \
            "Failed with batch_size=1"
        assert batch['mel'].shape == (1, 16, 80, 32), \
            "Failed mel shape with batch_size=1"
        
        # Test with different seeds
        batch1 = generate_dummy_batch(batch_size=4, seed=42)
        batch2 = generate_dummy_batch(batch_size=4, seed=42)
        assert torch.allclose(batch1['frames'], batch2['frames']), \
            "Same seed produced different visual data"
        assert torch.allclose(batch1['mel'], batch2['mel']), \
            "Same seed produced different audio data"
        
        # Test dataset indexing
        dataset = BasicDataset(num_samples=10, use_dummy_data=True)
        try:
            dataset[100]  # Out of bounds
            assert False, "Should raise IndexError"
        except IndexError:
            pass  # Expected
        
        print("    ✅ Edge cases PASSED")
        return True
    
    except Exception as e:
        print(f"    ❌ Edge cases FAILED: {e}")
        return False


def main():
    """Run all Phase 1 tests"""
    print_header("PHASE 1: VALIDATION TEST SUITE", 80)
    
    results = []
    results.append(("Dummy Data Generation", test_dummy_data_generation()))
    results.append(("Batch Shape Validation", test_batch_shape_validation()))
    results.append(("Balanced Labels", test_balanced_labels()))
    results.append(("BasicDataset Class", test_dataset_class()))
    results.append(("DataLoader", test_dataloader()))
    results.append(("Edge Cases", test_edge_cases()))
    
    # Summary
    print_header("TEST SUMMARY", 80)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<40} {status}")
    
    print("\n" + "-" * 80)
    print(f"Total: {passed}/{total} tests passed")
    print("-" * 80)
    
    if passed == total:
        print("\n🎉 ALL PHASE 1 TESTS PASSED! Ready for Phase 2.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before proceeding.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)