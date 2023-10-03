# variational_pruning-napari-plugin

## About

This repo is a napari plugin that features variational pruning -- a new algorithm to prune medial axis in 2D. The plugin serves as a visualization tool to help users better view the output of the pruned medial axis. The plugin also allows user to compare variational pruning's performance against an existing classic pruning metric (Erosion Thickness).

This repo is also the code repo for paper: Variational Pruning of Medial Axes of Planar Shapes (Eurographics Symposium on Geometry Processing 2023)

Raw algorithm code without visualization tool can be found at: https://github.com/peter-rong/dynamic_PCST

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

Use the "select image file" or "select curve file" button to load in the input data. Image file should be a gray-scale image file and curve file should be a txt file. (Sample input files can be found inside skeleton_plugin/sample_inputs.

After setting the wanted binary threshold or fairing number, choose Load Curve or Load Image based on your input file type. The boundary of the input file would be visible. Zoom and drag for best visualization purpose.

The color and prune button shows different visualization result (color shows the entire medial axis and prune shows the resulting medial axis under the given threshold) and modifying the slider would update the result in real time.

Once there's a color or prune visualization result, user can use the export button for the corresponding pruning method they selected and a output.txt file would be generated that records the resulting medial axis. 


### command line operation
```
cd skeleton_plugin
python3 execute.py <input_type> <input_file> <binary_threshold/fairing_number> <pruning_method> <pruning_threshhold> <visualization_method> <output_file>
```

```
<input_type> : "image" or "graph" 
<binary_threshold> : float between 0 and 255
<fairing_number>: non-negative integer
<pruning_method> : "VA" or "ET"
<pruning_threshhold> : float between 0 and 90 for VA ; float between 0 and 1 for ET
<visualization_method> : "color" or "prune"
```

Sample usage: 
```
python3 execute.py image toy.png 0 VA 45 prune output.txt
```

## Reference
This plugin is an extended work from the napari plugin at: https://github.com/teeli8/skeleton_plugin


