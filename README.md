# TrackR-CNN
Code for the TrackR-CNN baseline for the Multi Object Tracking and Segmentation (MOTS) task.

## Project website (including annotations)
https://www.vision.rwth-aachen.de/page/mots

## Paper
https://www.vision.rwth-aachen.de/media/papers/mots-multi-object-tracking-and-segmentation/MOTS.pdf

## mots_tools for evaluating results
https://github.com/VisualComputingInstitute/mots_tools

## Running this code
You'll need to install the following packages (possibly more):
```
tensorflow-gpu pycocotools numpy scipy sklearn pypng opencv-python munkres
```
Also create the following directories for logs, model files etc.:
```
mkdir forwarded models summaries logs
```
TODO

## References
Parts of this code are based on Tensorpack (https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN) and RETURNN (https://github.com/rwth-i6/returnn/blob/master/Log.py).

## Citation
If you use this code, please cite:
```
@inproceedings{Voigtlaender19CVPR_MOTS,
 author = {Paul Voigtlaender and Michael Krause and Aljo\u{s}a O\u{s}ep and Jonathon Luiten and Berin Balachandar Gnana Sekar and Andreas Geiger and Bastian Leibe},
 title = {{MOTS}: Multi-Object Tracking and Segmentation},
 booktitle = {CVPR},
 year = {2019},
}
```

## License
MIT License
