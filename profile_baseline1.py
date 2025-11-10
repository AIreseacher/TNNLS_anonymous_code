
import torch
import torch.nn as nn
import time
import copy
import numpy as np
from thop import profile as thop_profile
import argparse
import utils
from models import ResNet18WithFeatures

# ---------------------- 参数 ---------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="Path to dataset JSON file")
parser.add_argument("--model_path", type=str, required=True, help="Path to trained Baseline1 model")
parser.add_argument("--num_classes", type=int, default=5, help="Number of classes")
parser.add_argument("--batch_size", type=int, default=256, help="Batch size for testing")
args = parser.parse_args()

# ---------------------- Step 1: 加载数据 ---------------------- #
transform = utils.get_transforms()
_, _, test_loader = utils.create_data_loaders(
    json_file=args.data_path, transform=transform, batch_size=args.batch_size
)
num_images = len(test_loader.dataset)
print(f"[INFO] Test samples: {num_images}")

# ---------------------- Step 2: 加载模型 ---------------------- #
device_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18WithFeatures(num_classes=args.num_classes)
model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
model.eval()

# ---------------------- Step 3: Params & FLOPs ---------------------- #
sample = next(iter(test_loader))
sample_x = sample[0][0:1].contiguous()

params_m = sum(p.numel() for p in model.parameters()) / 1e6
flops, _ = thop_profile(copy.deepcopy(model), inputs=(sample_x,), verbose=False)
flops_g = flops / 1e9

# ---------------------- Step 4: GPU 平均推理时间（每张图片） ---------------------- #
if torch.cuda.is_available():
    model_gpu = copy.deepcopy(model).to(device_gpu).eval()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.float().to(device_gpu, non_blocking=True)
            _ = model_gpu(imgs)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    total_gpu_time = t1 - t0
    gpu_time_per_image = total_gpu_time / num_images
else:
    gpu_time_per_image = None

# ---------------------- Step 5: CPU 平均推理时间（每张图片） ---------------------- #
model_cpu = copy.deepcopy(model).to("cpu").eval()
t0 = time.perf_counter()
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.float().to("cpu")
        _ = model_cpu(imgs)
t1 = time.perf_counter()
total_cpu_time = t1 - t0
cpu_time_per_image = total_cpu_time / num_images

# ---------------------- Step 6: 性能评估 ---------------------- #
test_acc, test_macro_f1, test_auc_ovo, test_auc_ovr = utils.evaluate_model(
    model.to(device_gpu), test_loader, device_gpu, num_classes=args.num_classes
)

# ---------------------- Step 7: 打印结果 ---------------------- #
print("\n========== Baseline1 Profiling Results ==========")
print(f"Params (M):           {params_m:.2f}")
print(f"FLOPs  (G):           {flops_g:.2f}")
print(f"GPU Time / image (s): {gpu_time_per_image:.6f}" if gpu_time_per_image else "GPU not available")
print(f"CPU Time / image (s): {cpu_time_per_image:.6f}")
print(f"Test ACC:             {test_acc*100:.2f}%")
print(f"Test Macro-F1:        {test_macro_f1*100:.2f}%")
print(f"Test AUC-OVO:         {test_auc_ovo*100:.2f}%")
print(f"Test AUC-OVR:         {test_auc_ovr*100:.2f}%")
print("===============================================\n")
