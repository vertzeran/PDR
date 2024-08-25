import torch
from os.path import join
import ntpath
import os

if __name__ == '__main__':
    window_size = 250
    add_quat = True
    input_size = (1, 3, window_size, 1)
    if add_quat:
        input_size = (1, 7, window_size, 1)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_tensor = torch.randn(input_size).to(device)
    root_dir = '/home/maint/git/walking_direction_estimation/data/optimization_results/2023-03-13T11:54:13.910513optimization results on mixed, window size: 250_SZ_mixed_modes'
    file_name = 'WDE_regressor_mixed_LinAcc_0.329.pth'
    model_path = join(root_dir, file_name)
    res18model = torch.load(model_path)
    res18model.to(device)

    file_name_wo_extention = os.path.splitext(file_name)[0]
    trace_file_path = join(root_dir, file_name_wo_extention + '_trace.pt')
    model_in_trace_format = torch.jit.trace(res18model, input_tensor)
    model_in_trace_format.save(trace_file_path)

    onnx_file_path = os.path.join(root_dir, file_name_wo_extention + '.onnx')
    torch.onnx.export(res18model, input_tensor, onnx_file_path, export_params=True, opset_version=11,
                      do_constant_folding=True, input_names=["input"], output_names=["output"])