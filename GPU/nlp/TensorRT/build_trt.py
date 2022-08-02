import tensorrt as trt
from functools import partial

onnx_file_name = 'tf_bert.onnx'
tensorrt_file_name = 'bert.engine'
fp_16_mode = False
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


builder = trt.Builder(TRT_LOGGER)

config = builder.create_builder_config()
workspace=4
config.max_workspace_size = workspace * 1 << 30

flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
network = builder.create_network(flag)
parser = trt.OnnxParser(network, TRT_LOGGER)

if fp_16_mode : 
    config.set_flag(trt.BuilderFlag.FP16)
 
with open(onnx_file_name, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print (parser.get_error(error))
 
inputs = [network.get_input(i) for i in range(network.num_inputs)]
outputs = [network.get_output(i) for i in range(network.num_outputs)]
for inp in inputs:
   print(f'input "{inp.name}" with shape {inp.shape} and dtype {inp.dtype}')
for out in outputs:
    print(f'output "{out.name}" with shape {out.shape} and dtype {out.dtype}')

network.mark_output(network.get_layer(network.num_layers - 1).get_output(0))
engine = builder.build_serialized_network(network,config)
buf = engine.serialize()
with open(tensorrt_file_name, 'wb') as f:
    f.write(buf)