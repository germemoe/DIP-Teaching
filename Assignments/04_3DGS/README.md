# Assignment 4 - Implement Simplified 3D Gaussian Splatting



### Train
First, we use Colmap to recover camera poses and a set of 3D points. Please refer to [11-3D_from_Multiview.pptx](https://rec.ustc.edu.cn/share/705bfa50-6e53-11ef-b955-bb76c0fede49) to review the technical details.
```
python mvs_with_colmap.py --data_dir data/chair
```

Debug the reconstruction by running:
```
python debug_mvs_by_projecting_pts.py --data_dir data/chair
```

train 3dgs
```
python train.py --colmap_dir data/chair --checkpoint_dir data/chair/checkpoints
```

### Result
机位31在80次迭代的渲染图如下,loss在0.027左右


![r_31](data/chair/checkpoints/debug_images/epoch_0080/r_31.png)

debug视频如下

![debug_rendering](data/chair/checkpoints/debug_rendering.gif)
