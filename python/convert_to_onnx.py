import torch
import cnn

'''
okay easy script for once, just convert a model to onnx so it can be used in unity.
'''

file_in = "feedforwardnet234.pth"
file_out = "test_0_8000.onnx"

cnn = cnn.CNNNetwork(num_outputs=5)
state_dict = torch.load(f"../networks/{file_in}")
cnn.load_state_dict(state_dict)

shape = (1, 1, 64, 63)
x = torch.randn(shape)

torch.onnx.export(cnn,  # model being run
                  x,  # model input (or a tuple for multiple inputs)
                  f"../networks/onnx/{file_out}",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=9,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['X'],  # the model's input names
                  output_names=['Y'])
