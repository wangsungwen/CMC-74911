import os
from pathlib import Path

def fix_encoding(directory_path):
    print(f"--- Fixing encoding in {directory_path} ---")
    
    # Define a list to store files that couldn't be fixed (should be none with latin-1 read)
    failed_files = [] 

    for root, _, files in os.walk(directory_path):
        for file_name in files:
            if file_name.endswith(".txt"):
                file_path = Path(root) / file_name
                try:
                    # Read in binary mode to check for BOM bytes explicitly
                    with open(file_path, 'rb') as f:
                        content_bytes = f.read()
                    
                    # Check for and remove UTF-8 BOM (EF BB BF)
                    if content_bytes.startswith(b'\xef\xbb\xbf'):
                        content_bytes = content_bytes[3:] # Strip the first 3 bytes (BOM)
                    
                    # Decode to string and then write back as UTF-8 without BOM
                    content_str = content_bytes.decode('utf-8')
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content_str)
                    print(f"✅ Fixed encoding for: {file_name}")
                except Exception as e:
                    print(f"❌ Error fixing encoding for {file_name}: {e}")
                    failed_files.append(str(file_path))
    
    if not failed_files:
        print(f"--- All files in {directory_path} processed successfully. ---")
    else:
        print(f"--- Failed to process the following files in {directory_path}: ---")
        for f_path in failed_files:
            print(f_path)

if __name__ == "__main__":
    label_dirs_to_fix = [
        Path(r"datasets\wildlife_laser\trains\labels")
    ]

    for d in label_dirs_to_fix:
        if d.exists():
            fix_encoding(d)
        else:
            print(f"⚠️ Warning: Directory not found: {d}. Skipping.")
