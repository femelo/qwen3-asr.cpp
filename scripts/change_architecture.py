#!/usr/bin/env python3
from gguf import GGUFReader, GGUFWriter, GGUFValueType
import typer


def rename_arch(input_path: str, output_path: str, new_arch: str) -> None:
    print(f"* Loading: {input_path}")
    reader = GGUFReader(input_path)

    # Initialize the writer with the NEW architecture name
    writer = GGUFWriter(output_path, new_arch)
    
    # 1. Copy Metadata (excluding automatic/changed keys)
    skip_keys = ["general.architecture", "GGUF.version", "GGUF.tensor_count", "GGUF.kv_count"]
    
    for key, field in reader.fields.items():
        if key in skip_keys:
            continue
            
        vtype = field.types[0]
        
        # Handle different types for the add_key_value call
        if vtype == GGUFValueType.STRING:
            val = str(bytes(field.parts[-1]), encoding="utf-8")
            writer.add_key_value(key, val, vtype)
        elif vtype == GGUFValueType.ARRAY:
            sub_type = field.types[1]
            # Extract array items based on whether they are strings or numbers
            if sub_type == GGUFValueType.STRING:
                val = [str(bytes(field.parts[idx]), encoding="utf-8") for idx in field.data]
            else:
                val = [pv for idx in field.data for pv in field.parts[idx].tolist()]
            writer.add_key_value(key, val, vtype, sub_type=sub_type)
        else:
            # Simple scalars (INT, FLOAT, etc.)
            val = field.parts[-1][0]
            writer.add_key_value(key, val, vtype)

    # 2. Copy Tensors
    print(f"* Copying {len(reader.tensors)} tensors...")
    for tensor in reader.tensors:
        writer.add_tensor(
            tensor.name,
            tensor.data,
            raw_shape=list(map(int, tensor.shape)),
            raw_dtype=tensor.tensor_type,
        )

    print(f"* Writing to {output_path} as architecture: {new_arch}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print("Done!")


if __name__ == "__main__":
    typer.run(rename_arch)

# 1. Change qwen3-asr -> qwen2
# 2. Run llama-quantize
# 3. Change qwen2 -> qwen3-asr