import os

# ============================================================
# 删除非 best.pth 的 checkpoint 文件
# 路径: Y:\chentao\TNNLS\checkpoints
# ============================================================

root_dir = r"/nas/unas15/chentao/TNNLS/checkpoints"  # ← 修改为你的目标路径
delete_count = 0

for dirpath, dirnames, filenames in os.walk(root_dir):
    for fname in filenames:
        # 只处理以 .pth 结尾的文件
        if fname.endswith(".pth"):
            # 如果文件名不是 best.pth → 删除
            if fname != "best.pth":
                full_path = os.path.join(dirpath, fname)
                try:
                    os.remove(full_path)
                    print(f"[删除] {full_path}")
                    delete_count += 1
                except Exception as e:
                    print(f"[跳过] {full_path} (原因: {e})")

print(f"\n✅ 清理完成，共删除 {delete_count} 个非 best.pth 文件。")
