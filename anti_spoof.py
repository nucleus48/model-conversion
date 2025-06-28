import ai_edge_torch
import torch
import tensorflow as tf
import numpy

from src.anti_spoof_predict import load_model

torch_model = load_model("./resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth", 0)
torch_model.eval()

torch_inputs = (torch.randn(1, 3, 80, 80),)
torch_output = torch_model(*torch_inputs)

edge_model = ai_edge_torch.convert(torch_model, torch_inputs)
edge_output = edge_model(*torch_inputs)

if numpy.allclose(
    torch_output.detach().numpy(),
    edge_output,
    atol=1e-5,
    rtol=1e-5,
):
    print("Inference result with Pytorch and TfLite was within tolerance")
else:
    print("Something wrong with Pytorch --> TfLite")

edge_model.export("./anti-spoof.tflite")
