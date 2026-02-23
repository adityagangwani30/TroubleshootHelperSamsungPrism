import shutil
src = "scripts/diag_results.txt"
dst = "scripts/diag_results.md"
with open(src, "r", encoding="utf-8") as f:
    content = f.read()
with open(dst, "w", encoding="utf-8") as f:
    f.write("```\n")
    f.write(content)
    f.write("```\n")
print("Copied to", dst)
