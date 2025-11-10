import os, torch

ckpt = "/nas/unas15/chentao/TNNLS/checkpoints/LUNG/ResNet101_ResNet18/scope/seed_1234/best.pth"
obj = torch.load(ckpt, map_location="cpu")

# 兼容不同保存方式：可能是 {'state_dict':..., 'optimizer':...} 或 {'model':...} 或 直接 state_dict
candidates = []
for k in ["state_dict", "model", "module", "net", "weights"]:
    if isinstance(obj, dict) and k in obj and isinstance(obj[k], dict):
        candidates.append(obj[k])
if not candidates and isinstance(obj, dict):
    # 有些直接就是 state_dict
    candidates.append(obj)

def is_student_key(name: str) -> bool:
    # 只保留“学生”相关权重；根据你项目的命名做白名单/黑名单
    # 例：排除 teacher / ema / optimizer 等
    bad = ["teacher", ".T.", "ema", "optimizer", "opt_state", "sched", "grad", "momentum", "adam", "scaler"]
    if any(b in name.lower() for b in bad):
        return False
    # 如果你的学生模块有清晰前缀（如 'student.', 'model.student.'），也可进一步收紧过滤：
    # return name.startswith("student.") or name.startswith("model.student.")
    return True

total_params = 0
student_params = 0
for sd in candidates:
    for name, tensor in sd.items():
        if not torch.is_tensor(tensor):
            continue
        n = tensor.numel()
        total_params += n
        if is_student_key(name):
            student_params += n

print(f"[Exact] total params in file: {total_params/1e6:.2f} M (all tensors in checkpoint)")
print(f"[Exact] student-only params:  {student_params/1e6:.2f} M (filtered)")
print(f"[Info ] file size on disk:    {os.path.getsize(ckpt)/(1024**2):.1f} MB")
