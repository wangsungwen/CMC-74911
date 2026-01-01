import os

directories = ["datasets/wildlife/labels/train_laser", "datasets/wildlife/labels/val_laser"]

for directory in directories:
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as f:
                lines = f.readlines()
            
            modified_lines = []
            for line in lines:
                if line.startswith('0 '):
                    modified_lines.append('1' + line[1:])
                else:
                    modified_lines.append(line)
            
            with open(filepath, 'w') as f:
                f.writelines(modified_lines)
    print(f"All '0' labels in {directory} have been changed to '1'.")

print("All specified directories have been processed.")
