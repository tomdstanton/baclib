
import numpy as np
import sys
from baclib.containers.seq import CompressedSeqBatch, CompressedSeq
from baclib.core.alphabet import Alphabet

def test_compressed_seq_batch_empty():
    print("Testing empty()...")
    try:
        batch = CompressedSeqBatch.empty()
    except AttributeError:
        print("FAILED: CompressedSeqBatch.empty() not implemented")
        return False
        
    assert isinstance(batch, CompressedSeqBatch)
    assert len(batch) == 0
    assert batch.alphabet == Alphabet.DNA  # Default
    # Accessing private member for verification is okay in tests
    assert batch._bits == 2 # Default
    print("PASSED")
    return True

def test_compressed_seq_batch_zeros():
    print("Testing zeros(10)...")
    n = 10
    try:
        batch = CompressedSeqBatch.zeros(n)
    except AttributeError:
        print("FAILED: CompressedSeqBatch.zeros() not implemented")
        return False

    assert isinstance(batch, CompressedSeqBatch)
    assert len(batch) == n
    assert batch.alphabet == Alphabet.DNA
    assert len(batch[0]) == 0
    print("PASSED")
    return True

def test_enforcement():
    print("Testing enforcement...")
    
    # Texture direct instantiation fails
    try:
        CompressedSeqBatch(np.empty(0, dtype=np.uint8), np.empty(0, dtype=np.int32), 
                           np.empty(0, dtype=np.int32), Alphabet.DNA, 2)
        print("FAILED: Direct instantiation should raise PermissionError")
        return False
    except PermissionError:
        print("Confirmed PermissionError on direct instantiation")
        
    # Test Alphabet factory
    batch = Alphabet.DNA.empty_compressed()
    assert isinstance(batch, CompressedSeqBatch)
    
    batch = Alphabet.DNA.zeros_compressed(5)
    assert len(batch) == 5
    
    print("PASSED")
    return True

if __name__ == "__main__":
    passed = True
    passed &= test_compressed_seq_batch_empty()
    passed &= test_compressed_seq_batch_zeros()
    passed &= test_enforcement()
    
    if not passed:
        sys.exit(1)
