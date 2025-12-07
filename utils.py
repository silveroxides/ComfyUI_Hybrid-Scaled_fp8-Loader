import os
import json
import torch
import struct
import argparse
import sys

def tensor_to_dict(tensor_data):
    byte_data = bytes(tensor_data.tolist())
    json_str = byte_data.decode('utf-8')
    data_dict = json.loads(json_str)
    return data_dict


class MemoryEfficientSafeOpen:
    def __init__(self, filename, device='cpu'):
        self.filename = filename
        self.device = device
        self.header, self.header_size = self._read_header()
        self.file = open(filename, "rb")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.file.close()

    def keys(self):
        return [k for k in self.header.keys() if k != "__metadata__"]

    def get_tensor(self, key):
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]

        tensor_bytes = None
        if offset_start != offset_end:
            self.file.seek(self.header_size + 8 + offset_start)
            tensor_bytes = self.file.read(offset_end - offset_start)

        return self._deserialize_tensor(tensor_bytes, metadata)
    
    def get_tensor_as_dict(self, key):
        """Get a uint8 tensor and convert it to a dictionary.
        
        Args:
            key: The tensor key to retrieve
            
        Returns:
            dict: The decoded dictionary from the uint8 tensor
            
        Raises:
            ValueError: If the tensor is not uint8 dtype
        """
        tensor = self.get_tensor(key)
        metadata = self.header[key]
        
        if metadata["dtype"] != "U8":
            raise ValueError(f"Tensor '{key}' has dtype {metadata['dtype']}, expected U8 (uint8)")
        
        return tensor_to_dict(tensor)

    def _read_header(self):
        with open(self.filename, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            header_json = f.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def _deserialize_tensor(self, tensor_bytes, metadata):
        dtype_str = metadata["dtype"]
        shape = metadata["shape"]
        dtype = self._get_torch_dtype(dtype_str)

        if tensor_bytes is None:
            byte_tensor = torch.empty(0, dtype=torch.uint8)
        else:
            byte_tensor = torch.frombuffer(bytearray(tensor_bytes), dtype=torch.uint8)

        if dtype_str in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, dtype_str, shape)

        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        dtype_map = {
            "F64": torch.float64, "F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16,
            "I64": torch.int64, "I32": torch.int32, "I16": torch.int16, "I8": torch.int8,
            "U8": torch.uint8, "BOOL": torch.bool,
        }
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn

        dtype = dtype_map.get(dtype_str)
        if dtype is None:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return dtype

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            raise ValueError(f"Unsupported float8 type: {dtype_str}. Your PyTorch version may be too old.")


def main():
    """Command-line interface for inspecting safetensors files."""
    parser = argparse.ArgumentParser(
        description="Memory-efficient safetensors file inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # List all tensor keys in a file
  python test-memeffsafeopen.py model.safetensors --list
  
  # List with shape and dtype information
  python test-memeffsafeopen.py model.safetensors --list --show-shape --show-dtype
  
  # Show info about a specific tensor
  python test-memeffsafeopen.py model.safetensors --tensor model.layers.0.weight
  
  # Get uint8 tensor as dictionary
  python test-memeffsafeopen.py model.safetensors --tensor config --as-dict
  
  # Show info about all tensors
  python test-memeffsafeopen.py model.safetensors --info
"""
    )
    parser.add_argument("filename", help="Path to the safetensors file")
    parser.add_argument("--list", action="store_true", help="List all tensor keys")
    parser.add_argument("--show-shape", action="store_true", help="Show tensor shapes (use with --list)")
    parser.add_argument("--show-dtype", action="store_true", help="Show tensor dtypes (use with --list)")
    parser.add_argument("--tensor", type=str, help="Show information about a specific tensor")
    parser.add_argument("--as-dict", action="store_true", help="Convert uint8 tensor to dict (use with --tensor)")
    parser.add_argument("--info", action="store_true", help="Show information about all tensors")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (default: cpu)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found", file=sys.stderr)
        return 1
    
    try:
        with MemoryEfficientSafeOpen(args.filename, device=args.device) as f:
            if args.list:
                keys = f.keys()
                print(f"Found {len(keys)} tensors:")
                for key in keys:
                    output = f"  {key}"
                    
                    if args.show_shape or args.show_dtype:
                        metadata = f.header[key]
                        details = []
                        
                        if args.show_shape:
                            details.append(f"shape={metadata['shape']}")
                        
                        if args.show_dtype:
                            details.append(f"dtype={metadata['dtype']}")
                        
                        output += f" ({', '.join(details)})"
                    
                    print(output)
            
            elif args.tensor:
                if args.as_dict:
                    try:
                        data_dict = f.get_tensor_as_dict(args.tensor)
                        print(f"Tensor '{args.tensor}' as dictionary:")
                        print(json.dumps(data_dict, indent=2))
                    except ValueError as e:
                        print(f"Error: {e}", file=sys.stderr)
                        return 1
                else:
                    tensor = f.get_tensor(args.tensor)
                    metadata = f.header[args.tensor]
                    print(f"Tensor: {args.tensor}")
                    print(f"  Shape: {tensor.shape}")
                    print(f"  Dtype: {metadata['dtype']} -> {tensor.dtype}")
                    print(f"  Size: {tensor.numel()} elements")
                    print(f"  Memory: {tensor.element_size() * tensor.numel() / (1024**2):.2f} MB")
                    
                    # If uint8, show hint about --as-dict
                    if metadata['dtype'] == 'U8':
                        print(f"  Hint: Use --as-dict to convert this uint8 tensor to a dictionary")
            
            elif args.info:
                keys = f.keys()
                print(f"Safetensors file: {args.filename}")
                print(f"Number of tensors: {len(keys)}")
                print()
                sys.stdout.flush()
                
                total_params = 0
                total_bytes = 0
                for key in keys:
                    metadata = f.header[key]
                    shape = metadata['shape']
                    dtype = metadata['dtype']
                    
                    # Calculate number of elements
                    num_params = 1
                    for dim in shape:
                        num_params *= dim
                    total_params += num_params
                    
                    # Calculate bytes based on dtype
                    dtype_sizes = {
                        'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
                        'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1,
                        'U8': 1, 'BOOL': 1,
                        'F8_E5M2': 1, 'F8_E4M3': 1
                    }
                    dtype_size = dtype_sizes.get(dtype, 4)
                    tensor_bytes = num_params * dtype_size
                    total_bytes += tensor_bytes
                    
                    print(f"{key}:")
                    print(f"  Shape: {shape}")
                    print(f"  Dtype: {dtype}")
                    print(f"  Elements: {num_params:,}")
                    print(f"  Memory: {tensor_bytes / (1024**2):.2f} MB")
                    print()
                    sys.stdout.flush()
                
                print(f"Total elements: {total_params:,}")
                print(f"Total memory: {total_bytes / (1024**2):.2f} MB ({total_bytes / (1024**3):.3f} GB)")
                sys.stdout.flush()
            
            else:
                parser.print_help()
                return 1
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def run_tests():
    """Run comprehensive tests for MemoryEfficientSafeOpen class."""
    import tempfile
    
    print("Running tests for MemoryEfficientSafeOpen...\n")
    
    def create_test_safetensor(filename, tensors_dict):
        """Create a minimal safetensors file for testing."""
        header = {}
        current_offset = 0
        tensor_data = bytearray()
        
        for name, tensor in tensors_dict.items():
            tensor_bytes = tensor.numpy().tobytes()
            offset_start = current_offset
            offset_end = current_offset + len(tensor_bytes)
            
            header[name] = {
                "dtype": _torch_dtype_to_str(tensor.dtype),
                "shape": list(tensor.shape),
                "data_offsets": [offset_start, offset_end]
            }
            
            tensor_data.extend(tensor_bytes)
            current_offset = offset_end
        
        header_json = json.dumps(header).encode("utf-8")
        header_size = len(header_json)
        
        with open(filename, "wb") as f:
            f.write(struct.pack("<Q", header_size))
            f.write(header_json)
            f.write(tensor_data)
    
    def _torch_dtype_to_str(dtype):
        """Convert PyTorch dtype to safetensors dtype string."""
        dtype_map = {
            torch.float64: "F64", torch.float32: "F32", torch.float16: "F16", torch.bfloat16: "BF16",
            torch.int64: "I64", torch.int32: "I32", torch.int16: "I16", torch.int8: "I8",
            torch.uint8: "U8", torch.bool: "BOOL",
        }
        return dtype_map[dtype]
    
    # Test 1: Basic tensor loading
    print("Test 1: Basic tensor loading")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        test_tensor = torch.randn(3, 4)
        create_test_safetensor(tmp_path, {"test_tensor": test_tensor})
        
        with MemoryEfficientSafeOpen(tmp_path) as f:
            loaded = f.get_tensor("test_tensor")
            assert torch.allclose(loaded, test_tensor), "Tensor values don't match"
            assert loaded.shape == test_tensor.shape, "Tensor shapes don't match"
        print("  ✓ Passed\n")
    finally:
        os.unlink(tmp_path)
    
    # Test 2: Multiple tensors
    print("Test 2: Multiple tensors")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        tensors = {
            "weight": torch.randn(10, 20),
            "bias": torch.randn(20),
            "scale": torch.tensor([1.5]),
        }
        create_test_safetensor(tmp_path, tensors)
        
        with MemoryEfficientSafeOpen(tmp_path) as f:
            keys = f.keys()
            assert len(keys) == 3, f"Expected 3 keys, got {len(keys)}"
            assert set(keys) == set(tensors.keys()), "Keys don't match"
            
            for key in keys:
                loaded = f.get_tensor(key)
                assert torch.allclose(loaded, tensors[key]), f"Tensor {key} values don't match"
        print("  ✓ Passed\n")
    finally:
        os.unlink(tmp_path)
    
    # Test 3: Different dtypes
    print("Test 3: Different dtypes")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        tensors = {
            "float32": torch.randn(5, 5, dtype=torch.float32),
            "float16": torch.randn(5, 5, dtype=torch.float16),
            "int32": torch.randint(0, 100, (5, 5), dtype=torch.int32),
            "bool": torch.randint(0, 2, (5, 5), dtype=torch.bool),
        }
        create_test_safetensor(tmp_path, tensors)
        
        with MemoryEfficientSafeOpen(tmp_path) as f:
            for key, original in tensors.items():
                loaded = f.get_tensor(key)
                assert loaded.dtype == original.dtype, f"Dtype mismatch for {key}"
                assert torch.equal(loaded, original), f"Values don't match for {key}"
        print("  ✓ Passed\n")
    finally:
        os.unlink(tmp_path)
    
    # Test 4: Context manager
    print("Test 4: Context manager")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        create_test_safetensor(tmp_path, {"test": torch.randn(2, 2)})
        
        f = MemoryEfficientSafeOpen(tmp_path)
        f.__enter__()
        assert not f.file.closed, "File should be open"
        f.__exit__(None, None, None)
        assert f.file.closed, "File should be closed after exit"
        print("  ✓ Passed\n")
    finally:
        os.unlink(tmp_path)
    
    # Test 5: KeyError for missing tensor
    print("Test 5: KeyError for missing tensor")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        create_test_safetensor(tmp_path, {"existing": torch.randn(2, 2)})
        
        with MemoryEfficientSafeOpen(tmp_path) as f:
            try:
                f.get_tensor("nonexistent")
                assert False, "Should have raised KeyError"
            except KeyError as e:
                assert "not found" in str(e), "KeyError message incorrect"
        print("  ✓ Passed\n")
    finally:
        os.unlink(tmp_path)
    
    # Test 6: Empty tensor
    print("Test 6: Empty tensor")
    with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        empty_tensor = torch.empty(0, dtype=torch.float32)
        create_test_safetensor(tmp_path, {"empty": empty_tensor})
        
        with MemoryEfficientSafeOpen(tmp_path) as f:
            loaded = f.get_tensor("empty")
            assert loaded.shape == torch.Size([0]), "Empty tensor shape incorrect"
            assert loaded.numel() == 0, "Empty tensor should have 0 elements"
        print("  ✓ Passed\n")
    finally:
        os.unlink(tmp_path)
    
    print("All tests passed! ✓")


if __name__ == "__main__":
    # Check if running tests
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_tests()
    else:
        sys.exit(main())
