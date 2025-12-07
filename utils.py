import os
import json
import torch
import struct
import argparse
import sys
import re

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


class CallableMemEffSafeOpen:
    """Callable wrapper for MemoryEfficientSafeOpen with high-level operations.

    Provides convenient methods for listing, querying, and extracting tensor information
    from safetensors files without needing command-line arguments.
    """

    def __init__(self, filename, device='cpu'):
        """Initialize the callable interface.

        Args:
            filename: Path to the safetensors file
            device: Device to use (default: 'cpu')
        """
        self.filename = filename
        self.device = device
        self._safe_open = None

    def __enter__(self):
        self._safe_open = MemoryEfficientSafeOpen(self.filename, self.device)
        self._safe_open.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._safe_open:
            return self._safe_open.__exit__(exc_type, exc_val, exc_tb)

    def list_keys(self, show_shape=False, show_dtype=False):
        """List all tensor keys with optional metadata.

        Args:
            show_shape: Include tensor shapes
            show_dtype: Include tensor dtypes

        Returns:
            list: If no options, returns list of keys
            list of dict: If options enabled, returns list of dicts with metadata
        """
        keys = self._safe_open.keys()

        if not show_shape and not show_dtype:
            return keys

        results = []
        for key in keys:
            metadata = self._safe_open.header[key]
            result = {'key': key}

            if show_shape:
                result['shape'] = metadata['shape']

            if show_dtype:
                dtype_str = metadata['dtype']
                torch_dtype_map = {
                    'F64': 'torch.float64', 'F32': 'torch.float32',
                    'F16': 'torch.float16', 'BF16': 'torch.bfloat16',
                    'I64': 'torch.int64', 'I32': 'torch.int32',
                    'I16': 'torch.int16', 'I8': 'torch.int8',
                    'U8': 'torch.uint8', 'BOOL': 'torch.bool',
                    'F8_E5M2': 'torch.float8_e5m2', 'F8_E4M3': 'torch.float8_e4m3fn'
                }
                result['dtype'] = torch_dtype_map.get(dtype_str, dtype_str)

            results.append(result)

        return results

    def get_tensor_info(self, pattern, as_dict=False):
        """Get information about tensor(s) matching a pattern.

        Args:
            pattern: Regex pattern to match tensor keys (# is replaced with \\d)
            as_dict: If True, return uint8 tensors as dictionaries

        Returns:
            dict or list of dict: Tensor information or decoded dictionaries
        """
        # Replace # with \d for easier digit matching
        pattern_str = pattern.replace('#', r'\d')
        regex = re.compile(pattern_str)

        # Find all matching keys
        all_keys = self._safe_open.keys()
        matching_keys = [key for key in all_keys if regex.search(key)]

        if not matching_keys:
            return None

        if as_dict:
            # Return decoded uint8 tensors
            result = {}
            for key in matching_keys:
                try:
                    result[key] = self._safe_open.get_tensor_as_dict(key)
                except ValueError:
                    pass  # Skip non-uint8 tensors
            return result if result else None
        else:
            # Return tensor metadata
            results = []
            for key in matching_keys:
                tensor = self._safe_open.get_tensor(key)
                metadata = self._safe_open.header[key]

                results.append({
                    'key': key,
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype),
                    'dtype_raw': metadata['dtype'],
                    'elements': tensor.numel(),
                    'memory_mb': tensor.element_size() * tensor.numel() / (1024**2)
                })

            return results if len(results) > 1 else results[0]

    def get_tensor(self, key):
        """Get a tensor by key.

        Args:
            key: Tensor key

        Returns:
            torch.Tensor: The tensor
        """
        return self._safe_open.get_tensor(key)

    def get_tensor_as_dict(self, key):
        """Get a uint8 tensor as a dictionary.

        Args:
            key: Tensor key

        Returns:
            dict: Decoded dictionary from uint8 tensor
        """
        return self._safe_open.get_tensor_as_dict(key)

    def get_info(self):
        """Get comprehensive information about all tensors.

        Returns:
            dict: Summary with file info, tensor details, and totals
        """
        keys = self._safe_open.keys()

        tensor_infos = []
        total_elements = 0
        total_bytes = 0

        dtype_sizes = {
            'F64': 8, 'F32': 4, 'F16': 2, 'BF16': 2,
            'I64': 8, 'I32': 4, 'I16': 2, 'I8': 1,
            'U8': 1, 'BOOL': 1,
            'F8_E5M2': 1, 'F8_E4M3': 1
        }

        for key in keys:
            metadata = self._safe_open.header[key]
            shape = metadata['shape']
            dtype = metadata['dtype']

            # Calculate number of elements
            num_elements = 1
            for dim in shape:
                num_elements *= dim

            dtype_size = dtype_sizes.get(dtype, 4)
            tensor_bytes = num_elements * dtype_size

            total_elements += num_elements
            total_bytes += tensor_bytes

            tensor_infos.append({
                'key': key,
                'shape': shape,
                'dtype': dtype,
                'elements': num_elements,
                'memory_mb': tensor_bytes / (1024**2)
            })

        return {
            'filename': self.filename,
            'num_tensors': len(keys),
            'tensors': tensor_infos,
            'total_elements': total_elements,
            'total_memory_mb': total_bytes / (1024**2),
            'total_memory_gb': total_bytes / (1024**3)
        }


def main():
    """Command-line interface for inspecting safetensors files."""
    parser = argparse.ArgumentParser(
        description="Memory-efficient safetensors file inspector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  # List all tensor keys in a file
  python test-memeffsafeopen.py model.safetensors --list

  # List with shape and dtype information
  python test-memeffsafeopen.py model.safetensors --list dtype shape

  # Show info about a specific tensor
  python test-memeffsafeopen.py model.safetensors --tensor "model.layers.0.weight"

  # Show info about all tensors matching a pattern
  python test-memeffsafeopen.py model.safetensors --tensor "layers.*weight"

  # Use # as shorthand for \d (digit)
  python test-memeffsafeopen.py model.safetensors --tensor "layers.#.weight"

  # Get uint8 tensors as dictionary
  python test-memeffsafeopen.py model.safetensors --tensor "config" --as-dict

  # Show info about all tensors
  python test-memeffsafeopen.py model.safetensors --info
"""
    )
    parser.add_argument("filename", help="Path to the safetensors file")
    parser.add_argument("--list", nargs="*", choices=["dtype", "shape"], metavar="OPTION",
                       help="List all tensor keys. Optional: dtype, shape to show additional info")
    parser.add_argument("--tensor", type=str, help="Show information about tensor(s) matching regex pattern")
    parser.add_argument("--as-dict", action="store_true", help="Convert uint8 tensor(s) to dict (use with --tensor)")
    parser.add_argument("--info", action="store_true", help="Show information about all tensors")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (default: cpu)")

    args = parser.parse_args()

    if not os.path.exists(args.filename):
        print(f"Error: File '{args.filename}' not found", file=sys.stderr)
        return 1

    try:
        with MemoryEfficientSafeOpen(args.filename, device=args.device) as f:
            # Handle --list first (if provided)
            if args.list is not None:
                keys = f.keys()
                if not args.list:
                    print(f"Found {len(keys)} tensors:")
                    for key in keys:
                        print(f"  {key}")
                else:
                    # CSV format with header
                    header_parts = ["layer"]
                    if "shape" in args.list:
                        header_parts.append("shape")
                    if "dtype" in args.list:
                        header_parts.append("dtype")
                    print(", ".join(header_parts))

                    # CSV data rows
                    for key in keys:
                        metadata = f.header[key]
                        parts = [key]

                        if "shape" in args.list:
                            parts.append(str(metadata['shape']))

                        if "dtype" in args.list:
                            # Map dtype string to torch dtype representation
                            dtype_str = metadata['dtype']
                            torch_dtype_map = {
                                'F64': 'torch.float64', 'F32': 'torch.float32',
                                'F16': 'torch.float16', 'BF16': 'torch.bfloat16',
                                'I64': 'torch.int64', 'I32': 'torch.int32',
                                'I16': 'torch.int16', 'I8': 'torch.int8',
                                'U8': 'torch.uint8', 'BOOL': 'torch.bool',
                                'F8_E5M2': 'torch.float8_e5m2', 'F8_E4M3': 'torch.float8_e4m3fn'
                            }
                            parts.append(torch_dtype_map.get(dtype_str, dtype_str))

                        print(", ".join(parts))

                # If --info is also provided, print a separator and continue
                if args.info:
                    print()

            # Handle --tensor
            if args.tensor:
                # Replace # with \d for easier digit matching
                pattern_str = args.tensor.replace('#', r'\d')

                # Compile regex pattern
                try:
                    pattern = re.compile(pattern_str)
                except re.error as e:
                    print(f"Error: Invalid regex pattern: {e}", file=sys.stderr)
                    return 1

                # Find all matching keys
                all_keys = f.keys()
                matching_keys = [key for key in all_keys if pattern.search(key)]

                if not matching_keys:
                    print(f"No tensors matching pattern '{args.tensor}'", file=sys.stderr)
                    return 1

                if args.as_dict:
                    # Output all matching uint8 tensors as nested dict
                    result_dict = {}
                    for key in matching_keys:
                        try:
                            result_dict[key] = f.get_tensor_as_dict(key)
                        except ValueError as e:
                            print(f"Warning: Skipping '{key}': {e}", file=sys.stderr)

                    if result_dict:
                        print(json.dumps(result_dict, indent=2))
                    else:
                        print(f"Error: No uint8 tensors found matching pattern", file=sys.stderr)
                        return 1
                else:
                    # Output info for all matching tensors
                    for i, key in enumerate(matching_keys):
                        if i > 0:
                            print()

                        tensor = f.get_tensor(key)
                        metadata = f.header[key]
                        print(f"Tensor: {key}")
                        print(f"  Shape: {tensor.shape}")
                        print(f"  Dtype: {metadata['dtype']} -> {tensor.dtype}")
                        print(f"  Size: {tensor.numel()} elements")
                        print(f"  Memory: {tensor.element_size() * tensor.numel() / (1024**2):.2f} MB")

                        # If uint8, show hint about --as-dict
                        if metadata['dtype'] == 'U8':
                            print(f"  Hint: Use --as-dict to convert this uint8 tensor to a dictionary")

            # Handle --info
            if args.info:
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

            # If no action was taken, show help
            if args.list is None and not args.tensor and not args.info:
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


# # Example usage of CallableMemEffSafeOpen class
# # Uncomment to run these examples
#
# if __name__ == "__main__":
#     # Example 1: List all tensor keys
#     with CallableMemEffSafeOpen("model.safetensors") as f:
#         keys = f.list_keys()
#         print(f"Found {len(keys)} tensors:")
#         for key in keys:
#             print(f"  {key}")
#
#     # Example 2: List keys with shape and dtype
#     with CallableMemEffSafeOpen("model.safetensors") as f:
#         detailed = f.list_keys(show_shape=True, show_dtype=True)
#         for item in detailed:
#             print(f"{item['key']}: {item['shape']} - {item['dtype']}")
#
#     # Example 3: Get info for tensors matching a pattern
#     with CallableMemEffSafeOpen("model.safetensors") as f:
#         # Use # as shorthand for \d (digit)
#         info = f.get_tensor_info("layers.#.weight")
#         if isinstance(info, list):
#             for tensor_info in info:
#                 print(f"{tensor_info['key']}: {tensor_info['shape']}, {tensor_info['memory_mb']:.2f} MB")
#         else:
#             print(f"{info['key']}: {info['shape']}, {info['memory_mb']:.2f} MB")
#
#     # Example 4: Get all tensors matching a wildcard pattern
#     with CallableMemEffSafeOpen("model.safetensors") as f:
#         all_weights = f.get_tensor_info(".*weight")
#         print(f"Found {len(all_weights)} weight tensors")
#
#     # Example 5: Load a specific tensor
#     with CallableMemEffSafeOpen("model.safetensors") as f:
#         tensor = f.get_tensor("model.layers.0.weight")
#         print(f"Loaded tensor with shape: {tensor.shape}")
#
#     # Example 6: Get uint8 tensors as dictionaries
#     with CallableMemEffSafeOpen("model.safetensors") as f:
#         config_dicts = f.get_tensor_info("config.*", as_dict=True)
#         if config_dicts:
#             for key, value in config_dicts.items():
#                 print(f"{key}: {value}")
#
#     # Example 7: Get comprehensive info about all tensors
#     with CallableMemEffSafeOpen("model.safetensors") as f:
#         info = f.get_info()
#         print(f"File: {info['filename']}")
#         print(f"Total tensors: {info['num_tensors']}")
#         print(f"Total elements: {info['total_elements']:,}")
#         print(f"Total memory: {info['total_memory_gb']:.3f} GB")
#         print("\nTensor details:")
#         for tensor in info['tensors']:
#             print(f"  {tensor['key']}: {tensor['shape']} - {tensor['memory_mb']:.2f} MB")
#
#     # Example 8: Multiple operations in one context
#     with CallableMemEffSafeOpen("model.safetensors") as f:
#         # Get all keys
#         all_keys = f.list_keys()
#
#         # Get info for specific layers
#         layer_0 = f.get_tensor_info("layers.0.*")
#
#         # Load specific tensor
#         weight = f.get_tensor("model.weight")
#
#         # Get summary
#         summary = f.get_info()
#
#         print(f"Processed {len(all_keys)} tensors, loaded {weight.shape}")
