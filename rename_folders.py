# import os
# import shutil

# def rename_folders(root_dir, prefix='ina'):
#     # Get all subdirectories in the root directory
#     subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
#     # Sort the subdirectories alphabetically
#     subdirs.sort()
    
#     # Rename each subdirectory
#     for i, subdir in enumerate(subdirs, start=1):
#         old_path = os.path.join(root_dir, subdir)
#         new_name = f"{prefix}_{i:03d}_{subdir.replace(' ', '_')}"
#         new_path = os.path.join(root_dir, new_name)
        
#         # Rename the directory
#         os.rename(old_path, new_path)
#         print(f"Renamed: {subdir} -> {new_name}")

# # Usage
# root_directory = '/fastscratch/ksmehrab/INaturalist/INatBirdForHCompNet/Images/val'
# rename_folders(root_directory)

import os
import errno

def create_symlinks(root_dir, target_dir, prefix='ina'):
    # Ensure the target directory exists
    os.makedirs(target_dir, exist_ok=True)

    # Get all subdirectories in the root directory
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Sort the subdirectories alphabetically
    subdirs.sort()
    
    # Create symlink for each subdirectory
    for i, subdir in enumerate(subdirs, start=1):
        old_path = os.path.join(root_dir, subdir)
        new_name = f"{prefix}_{i:03d}_{subdir.replace(' ', '_')}"
        new_path = os.path.join(target_dir, new_name)
        
        try:
            os.symlink(old_path, new_path)
            print(f"Created symlink: {new_name} -> {old_path}")
        except OSError as e:
            if e.errno == errno.EEXIST:
                print(f"Symlink already exists: {new_name}")
            else:
                print(f"Error creating symlink for {subdir}: {e}")

# Usage
root_directory = '/fastscratch/ksmehrab/INaturalist/INatBirdForHCompNet/Images/train'
target_directory = '/fastscratch/ksmehrab/INaturalist/INatBirdForHCompNet/Images_symlink/train'
os.makedirs(target_directory, exist_ok=True)
create_symlinks(root_directory, target_directory)