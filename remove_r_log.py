import os

# Root directory
root_dir = "/nas/unas15/chentao/TNNLS/checkpoints3"

# Counter for deleted files
count = 0

# Walk through all subdirectories
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if filename == "r_log.log":
            file_path = os.path.join(dirpath, filename)
            try:
                os.remove(file_path)
                count += 1
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")

print(f"\n[INFO] Total deleted files: {count}")
