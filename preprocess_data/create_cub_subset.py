# Renames and copies only the species from phylogeny
import os
import shutil
import ete3
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--path',
                    type=str,
                    default='data/CUB_200_2011/images',
                    help='Path to CUB-200-2011 dataset')
parser.add_argument('--phylogeny_path',
                    type=str,
                    default='data/phlyogenyCUB/1_tree-consensus-Hacket-AllSpecies-modified_cub-names_v1.phy',
                    help='Path to cub190 phlyogeny dataset')
parser.add_argument('--target_path',
                    type=str,
                    default='data/CUB_200_2011/images_cub190',
                    help='Path to cub190 phlyogeny dataset')

                    
args = parser.parse_args()
path = args.path
phylogeny_path = args.phylogeny_path

tree = ete3.Tree(args.phylogeny_path)
leaf_names = [leaf.name for leaf in tree.get_leaves()]
folder_names = leaf_names

os.makedirs(args.target_path, exist_ok=True)

copied_count = 0
for folder in tqdm(os.listdir(args.path), total=len(os.listdir(args.path))):
    # target folder name in the same format as in the cub phylogeny
    target_folder = 'cub_' + folder[:3] + '_' + folder[4:]
    if target_folder in leaf_names:
        folder_path = os.path.join(args.path, folder)
        if os.path.isdir(folder_path):
            shutil.copytree(folder_path, os.path.join(args.target_path, target_folder))
            copied_count += 1
print(f'Copied {copied_count} folders')

