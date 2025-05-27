import ai_edge_torch
import numpy
import torch

torch_model = torch.hub.load(
    "otroshi/edgeface", "edgeface_xs_gamma_06", source="github", pretrained=True
)

torch_model.eval()

torch_inputs = (torch.randn(1, 3, 112, 112),)
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

edge_model.export("face-recognition.tflite")
