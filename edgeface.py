import ai_edge_torch
import torch
import tensorflow as tf

torch_model = torch.hub.load(
    "otroshi/edgeface", "edgeface_xs_gamma_06", source="github", pretrained=True
)

torch_inputs = (torch.randn(1, 3, 112, 112),)
tfl_converter_flags = {'optimizations': [tf.lite.Optimize.DEFAULT]}
edge_model = ai_edge_torch.convert(torch_model.eval(), torch_inputs, _ai_edge_converter_flags=tfl_converter_flags)
edge_model.export("./edgeface-xs.tflite")
