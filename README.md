# EasyMesh
Implementation of the article "EasyMesh: An Efficient Method to Reconstruct 3D Mesh From a Single Image "

## Prepare your data
1. Silhouette iamges: you can render [depth images](https://github.com/tasx0823/render-depth-image) first, and then conver depth iamge to silhouette images.
2. [Geometry image](https://github.com/sinhayan/surfnet)

### Model Training
```
python main_folding.py --phase train
```

### Model Testing
```
python main_folding.py --phase test
```
