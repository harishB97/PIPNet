from scripts.import_utils import instantiate_from_config
import scripts.constants as CONSTANTS
import torch.nn as nn
import torch
from torchmetrics import F1Score


import numpy as np

#NOTE: These are distances from tree root. If you want from leaves, you need to call get_relative_distance_for_level 
def parse_phyloDistances(phyloDistances_string):
    phyloDistances_list_string = phyloDistances_string.split(",")
    sorted_distance = sorted(list(map(lambda x: float(x), phyloDistances_list_string)))
    return sorted_distance

def get_relative_distance_for_level(phylo_distances, level):
    return 1.0- (phylo_distances[level] if level < len(phylo_distances) else 1.0)

def get_loss_name(phylo_distances, level):
    return str(phylo_distances[level]).replace(".", "")+"distance"

class Species_sibling_finder():
    # Contructor
    def __init__(self, phylogeny, genetic_distances_from_root):
        self.map = {}
        self.phylogeny = phylogeny
        for species in phylogeny.node_ids:
            self.map[species] = {}
            for indx, distance in enumerate(genetic_distances_from_root):
                distance_relative = get_relative_distance_for_level(genetic_distances_from_root, indx)
                self.map[species][get_loss_name(genetic_distances_from_root, indx)] = phylogeny.get_siblings_by_name(species, distance_relative)


    def map_speciesId_siblingVector(self, speciesId, loss_name):
        label_list = self.phylogeny.getLabelList()
        species = label_list[speciesId]
        siblings = self.map[species][loss_name]
        siblings_indices = list(map(lambda x: label_list.index(x), siblings))
        return siblings_indices
    
###----------------------------------###





class PhyloLoss(nn.Module):
    def __init__(self, phyloDistances_string, phylogenyconfig, phylo_weight, fc_layers=None, beta=1.0, verbose=False):
        super().__init__()
        self.phylo_distances = parse_phyloDistances(phyloDistances_string) 
        self.phylogeny = instantiate_from_config(phylogenyconfig)
        self.siblingfinder = Species_sibling_finder(self.phylogeny, self.phylo_distances)
        self.phylo_weight = phylo_weight
        self.fc_layers = fc_layers
        self.verbose = verbose
        self.beta = beta
        
        self.classifier_output_sizes = self.get_classification_output_sizes()
        
        self.mlb = {}
        for level, i in enumerate(self.phylo_distances):
            relative_distance = self.get_relative_distance_for_level(level)
            species_groups = self.phylogeny.get_species_groups(relative_distance, self.verbose)
            species_groups_representatives = list(map(lambda x: x[0], species_groups))
            species_groups_representatives = list(map(lambda x: self.phylogeny.getLabelList().index(x), species_groups_representatives))
            self.mlb[get_loss_name(self.phylo_distances, level)] = species_groups_representatives
                

        self.criterionCE = torch.nn.CrossEntropyLoss()
         
        self.F1 = F1Score(num_classes=self.classifier_output_sizes[-1], multiclass=True)
    
    def get_classification_output_sizes(self):   
        output_sizes = []
        for level, i in enumerate(self.phylo_distances):
            relative_distance = self.get_relative_distance_for_level(level)
            output_sizes.append(len(self.phylogeny.get_species_groups(relative_distance, self.verbose))) 
        output_sizes.append(len(self.phylogeny.getLabelList()))
        return output_sizes

    
    def get_relative_distance_for_level(self, level):
        return get_relative_distance_for_level(self.phylo_distances, level)
    
    def forward(self, cumulative_loss, activations, labels):
        losses_dict = {'individual_losses': {'class_loss': self.criterionCE(activations[CONSTANTS.DISENTANGLER_CLASS_OUTPUT], labels)}}

        for loss_name, activation in activations.items():
            if loss_name in CONSTANTS.CLASS_TENSORS:
                continue
            
            layer_truth = list(map(lambda x: self.siblingfinder.map_speciesId_siblingVector(x, loss_name), labels))
            ancestor_truth = torch.LongTensor(list(map(lambda x: self.mlb[loss_name].index(x[0]), layer_truth))).to(activation.device)
            losses_dict['individual_losses'][loss_name+"_loss"] = self.criterionCE(activation, ancestor_truth)


        # aggregate the losses
        # NOTE: Don't add losses you don't want to add up before here!
        total_phylo_loss = torch.stack(list(losses_dict['individual_losses'].values())).sum()
        total_phylo_loss = total_phylo_loss*self.beta + losses_dict['individual_losses']['class_loss']*(1-self.beta)
        losses_dict['total_phylo_loss'] = total_phylo_loss
        losses_dict['cumulative_loss'] = cumulative_loss + self.phylo_weight*total_phylo_loss

        class_f1 = self.F1(activations[CONSTANTS.DISENTANGLER_CLASS_OUTPUT], labels)
        losses_dict['class_f1'] = class_f1

        # return loss_dic
        return losses_dict
