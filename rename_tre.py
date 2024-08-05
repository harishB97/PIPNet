import os
from ete3 import Tree

def get_name_mapping(symlink_dir, prefix='ina'):
    name_mapping = {}
    for new_name in os.listdir(symlink_dir):
        if os.path.isdir(os.path.join(symlink_dir, new_name)) and new_name.startswith(prefix):
            parts = new_name.split('_', 2)
            if len(parts) == 3:
                old_name = parts[2].replace('_', ' ')
                name_mapping[old_name] = new_name
    return name_mapping

def update_tree_file(tree_file_path, name_mapping, output_tree_file_path):
    # Read the tree
    tree = Tree(tree_file_path, format=1)

    # Update leaf names
    for leaf in tree.iter_leaves():
        if ' '.join(leaf.name.split('_')) in name_mapping:
            leaf.name = name_mapping[' '.join(leaf.name.split('_'))]
    
    # Write the updated tree
    tree.write(outfile=output_tree_file_path, format=1)
    print(f"Updated tree file saved to: {output_tree_file_path}")

# Usage
symlink_directory = '/projects/ml4science/FishVistaForHCompNet/Max30ImagesPerSpecies/Images/train'
tree_file_path = '/projects/ml4science/FishVistaForHCompNet/Max30ImagesPerSpecies/fv419_final_tree.tre'
output_tree_file_path = '/projects/ml4science/FishVistaForHCompNet/Max30ImagesPerSpecies/fv419_final_tree_renamed.tre'

# Get name mapping from existing symlinks
name_mapping = get_name_mapping(symlink_directory, prefix='fvi')

# print(name_mapping)

# Update tree file
update_tree_file(tree_file_path, name_mapping, output_tree_file_path)