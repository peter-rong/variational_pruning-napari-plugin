# variational_pruning-napari-plugin

## About

This repo is a napari plugin that features variational pruning -- a new algorithm to prune medial axis in 2D. The plugin serves as a visualization tool to help users better view the output of the pruned medial axis. The plugin also allows user to compare variational pruning's performance against an existing classic pruning metric (Erosion Thickness).

This repo is also the code repo for paper: Variational Pruning of Medial Axes of Planar Shapes (Eurographics Symposium on Geometry Processing 2023)

## Installment 

### local installment (recommended for developer)
While importing from Version Control, set the local repo name to be "skeleton-plugin". If this step is not done, you would need to reconfigure the plugin credentials.

Python 3.10.6 is recommended to use for locally configuring this plugin. 

A list of packages used are included in the requirements.txt file. 

Once all the requirements are met, cd into your local repo and use the following command "pip install e ." to install the local plugin into napari.

## Usage

### plugin usage

Use command "napari" to open the napari interface

You can access the plugin from napari's plugin dropdown, use the main widget for visualization purpose. 

<img width="1331" alt="git image1" src="https://github.com/peter-rong/variational_pruning-napari-plugin/assets/71267071/21b818e5-1b0d-409e-af83-f452cab3c35c">



## Reference
This plugin is an extended work from the skeleton-plugin at: https://github.com/teeli8/skeleton_plugin

