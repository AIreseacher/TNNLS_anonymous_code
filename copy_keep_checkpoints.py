#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 Excel 最后一列（ckpt_path）将需要保留的 seed 目录完整复制到新的根目录，
保持原有的相对路径结构（从 'checkpoints/' 之后开始）。

示例：ckpt_path = checkpoints/Chaoyang/ResNet101_ResNet18/nosra_nogcr/seed_3407/best.pth
复制：
  源：/nas/unas15/chentao/TNNLS/checkpoints2/Chaoyang/ResNet101_ResNet18/nosra_nogcr/seed_3407/**
  目标：/nas/unas15/chentao/TNNLS/checkpoints3/Chaoyang/ResNet101_ResNet18/nosra_nogcr/seed_3407/**
"""

import os
import sys
import shutil
from pathlib import Path
import pandas as pd

# ================== 用户配置 ==================
SRC_ROOT = Path("/nas/unas15/chentao/TNNLS/checkpoints2")   # 源根目录
DST_ROOT = Path("/nas/unas15/chentao/TNNLS/checkpoints3")   # 目标根目录
EXCEL_PATH = Path("./Chaoyang_results.xlsx")                # Excel 文件路径
LAST_COLUMN_INDEX = -1                                      # 使用最后一列
DRY_RUN = True                                              # 预演：True=只打印不复制
# =================================================


def parse_seed_dir_from_ckpt(ckpt_path: str) -> Path | None:
    """
    从 Excel 的 ckpt_path 解析出 'seed_xxx' 所在目录的相对路径（相对于 checkpoints/ 之后）。
    返回示例：Path('Chaoyang/ResNet101_ResNet18/nosra_nogcr/seed_3407')
    """
    if not isinstance(ckpt_path, str):
        return None
    s = ckpt_path.strip().replace("\\", "/")
    if not s:
        return None
    # 去掉开头的 / 或 ./ 等
    while s.startswith("/"):
        s = s[1:]
    if s.startswith("./"):
        s = s[2:]
    parts = [p for p in s.split("/") if p]
    if not parts:
        return None

    # 去掉首段 'checkpoints'
    if parts[0].lower() == "checkpoints":
        parts = parts[1:]
    if len(parts) < 2:
        return None

    # 去掉最后的文件名（best.pth / last.pth / epoch_*.pth）
    parent_rel = parts[:-1]
    # parent_rel 应该以 seed_xxx 结尾；不做强校验，按上游数据可信处理
    return Path(*parent_rel)


def gather_keep_seed_dirs(excel_path: Path, last_col_idx: int) -> set[Path]:
    df = pd.read_excel(excel_path)
    series = df.iloc[:, last_col_idx].dropna().astype(str)
    keep_set: set[Path] = set()
    for val in series:
        rel_seed_dir = parse_seed_dir_from_ckpt(val)
        if rel_seed_dir is not None:
            keep_set.add(rel_seed_dir)
    return keep_set


def safe_copy_tree(src_dir: Path, dst_dir: Path, dry_run: bool = True) -> tuple[int, int]:
    """
    复制 src_dir 下的所有文件到 dst_dir（保持子结构）。
    返回：(copied_files, skipped_files)
    """
    copied = 0
    skipped = 0
    if not src_dir.exists() or not src_dir.is_dir():
        print(f"[WARN] 源目录不存在或不是目录：{src_dir}")
        return (0, 0)

    for root, dirs, files in os.walk(src_dir):
        root_p = Path(root)
        # 计算对应的目标目录
        rel = root_p.relative_to(src_dir)
        out_dir = dst_dir / rel
        if dry_run:
            print(f"[DRY-RUN][MKDIR] {out_dir}")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)

        for fn in files:
            src_file = root_p / fn
            dst_file = out_dir / fn
            if dry_run:
                print(f"[DRY-RUN][COPY] {src_file}  ->  {dst_file}")
                copied += 1
            else:
                try:
                    # copy2 保留元数据时间戳等
                    shutil.copy2(src_file, dst_file)
                    print(f"[COPY] {src_file}  ->  {dst_file}")
                    copied += 1
                except Exception as e:
                    print(f"[SKIP] {src_file}  (原因: {e})")
                    skipped += 1
    return (copied, skipped)


def main():
    # 基本检查
    if not EXCEL_PATH.exists():
        print(f"[ERROR] Excel 不存在：{EXCEL_PATH}")
        sys.exit(1)
    if not SRC_ROOT.exists():
        print(f"[ERROR] 源根目录不存在：{SRC_ROOT}")
        sys.exit(1)
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] 源根目录：{SRC_ROOT}")
    print(f"[INFO] 目标根目录：{DST_ROOT}")
    print(f"[INFO] Excel：{EXCEL_PATH}")
    print(f"[INFO] DRY_RUN：{DRY_RUN}")

    # 1) 收集要保留（复制）的 seed 目录（相对路径，基于 'checkpoints/' 之后）
    keep_rel_dirs = gather_keep_seed_dirs(EXCEL_PATH, LAST_COLUMN_INDEX)
    if not keep_rel_dirs:
        print("[WARN] Excel 中未解析到任何 ckpt 路径，结束。")
        sys.exit(0)

    print(f"[INFO] 解析到 {len(keep_rel_dirs)} 个 seed 目录：")
    for d in sorted(keep_rel_dirs):
        print(f"   - {d}")

    # 2) 逐个复制
    total_copied = 0
    total_skipped = 0
    missing = []
    for rel_dir in sorted(keep_rel_dirs):
        src_dir = (SRC_ROOT / rel_dir).resolve()
        dst_dir = (DST_ROOT / rel_dir).resolve()

        # 安全性：确保目标路径在目标根内
        try:
            dst_dir.relative_to(DST_ROOT.resolve())
        except Exception:
            print(f"[ERROR] 目标路径越界：{dst_dir}")
            continue

        if not src_dir.exists():
            print(f"[WARN] 缺失源目录：{src_dir}")
            missing.append(str(src_dir))
            continue

        copied, skipped = safe_copy_tree(src_dir, dst_dir, dry_run=DRY_RUN)
        total_copied += copied
        total_skipped += skipped

    print("\n================= 汇总 =================")
    print(f"[INFO] 计划复制文件（含 DRY-RUN）：{total_copied} 个")
    print(f"[INFO] 复制失败/跳过：{total_skipped} 个")
    if missing:
        print(f"[WARN] 缺失的源 seed 目录：{len(missing)} 个")
        for m in missing[:20]:
            print(f"   - {m}")
        if len(missing) > 20:
            print("   ...")

    print("\n✅ 完成（DRY_RUN 模式）" if DRY_RUN else "\n✅ 完成（已实际复制）")
    print("提示：确认无误后，将 DRY_RUN 置为 False 再次运行以执行真实复制。")


if __name__ == "__main__":
    main()
