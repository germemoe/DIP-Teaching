# Assignment 2 - DIP with PyTorch



### Requirements:

To install requirements:

```setup
pip install -r requirements.txt
```



### 1. Implement Poisson Image Editing with PyTorch.



因为没实现论文中的混合梯度所以前景区域没法很好的嵌入到背景中,会糊掉.



## Running

To run poisson image editing, run:

```basic
python run_blending_gradio.py
```



## Result

<img src="equation.png" alt="alt text" width="800">

<img src="monolisa.png" alt="alt text" width="800">

<img src="water.png" alt="alt text" width="800">

### 2. Pix2Pix implementation.
Assignments/02_DIPwithPyTorch/Pix2Pix/val_results/epoch_100


因为显卡限制所以批次改到了20.



## Training

run:

```bash
bash download_cityscapes_dataset.sh
python train.py
```



## Result

因为批次改到了20,而且数据集也换了稍大一些的cityscapes,所以下面只是迭代100次的结果示例,(loss因为关机了没保存,只有验证集图像)更多信息参考train_result和val_result,论文中用了个判别器和L1loss结合来构造loss,代码中只用了L1loss,并且也没迭代很多次,所以结果还是有挺多模糊的地方

<img src="Pix2Pix/val_results/epoch_100/result_1.png" alt="alt text" width="800">

<img src="Pix2Pix/val_results/epoch_100/result_2.png" alt="alt text" width="800">

<img src="Pix2Pix/val_results/epoch_100/result_3.png" alt="alt text" width="800">

<img src="Pix2Pix/val_results/epoch_100/result_4.png" alt="alt text" width="800">

<img src="Pix2Pix/val_results/epoch_100/result_5.png" alt="alt text" width="800">
