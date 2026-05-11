from pathlib import Path

root = Path("data/frames/normal")

identifier_files = list(root.rglob(".identifier"))

print(f"Found {len(identifier_files)} .identifier file(s):\n")

for file_path in identifier_files:
    print(file_path)

print("\nDry run only. No files were deleted.")