import onnx
from onnx import numpy_helper
import numpy as np

def convert_int64_to_int32(input_model_path, output_model_path):
    print(f"Loading ONNX model from {input_model_path}...")
    model = onnx.load(input_model_path)

    # 1. Convert Initializers (Weights and Constants)
    converted_initializers = 0
    for init in model.graph.initializer:
        if init.data_type == onnx.TensorProto.INT64:
            # Convert to numpy, cast to int32, and convert back to ONNX tensor
            np_arr = numpy_helper.to_array(init)
            np_arr = np_arr.astype(np.int32)
            
            new_init = numpy_helper.from_array(np_arr, init.name)
            
            # Replace old data
            init.data_type = onnx.TensorProto.INT32
            init.raw_data = new_init.raw_data
            converted_initializers += 1

    # 2. Convert Value Info (Inputs, Outputs, and intermediate tensors)
    def convert_type(value_info):
        count = 0
        if value_info.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            value_info.type.tensor_type.elem_type = onnx.TensorProto.INT32
            count += 1
        return count

    converted_io = 0
    for vi in model.graph.input: converted_io += convert_type(vi)
    for vi in model.graph.output: converted_io += convert_type(vi)
    for vi in model.graph.value_info: converted_io += convert_type(vi)

    # 3. Convert 'Cast' operation nodes
    # Some nodes explicitly tell the model to cast a tensor to INT64 (value 7 in ONNX proto). 
    # We change these to cast to INT32 (value 6 in ONNX proto).
    converted_nodes = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == onnx.TensorProto.INT64:
                    attr.i = onnx.TensorProto.INT32
                    converted_nodes += 1

    print(f"Converted {converted_initializers} initializers.")
    print(f"Converted {converted_io} inputs/outputs/value_infos.")
    print(f"Converted {converted_nodes} Cast nodes.")

    # Save the new model
    onnx.save(model, output_model_path)
    print(f"Successfully saved converted model to {output_model_path}")

if __name__ == "__main__":
    input_onnx = "best.onnx"
    output_onnx = "best_int32.onnx"
    convert_int64_to_int32(input_onnx, output_onnx)
