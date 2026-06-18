import torch

print('torch', torch.__version__)
print('cuda_version', torch.version.cuda)
print('cuda_available', torch.cuda.is_available())
print('device_count', torch.cuda.device_count())
if torch.cuda.is_available():
    try:
        print('device_name', torch.cuda.get_device_name(0))
    except Exception as e:
        print('device_name_error', e)
