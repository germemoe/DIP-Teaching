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

![equation](E:\USTC\DIP\DIP-Teaching\Assignments\02_DIPwithPyTorch\equation.png)

![monolisa](E:\USTC\DIP\DIP-Teaching\Assignments\02_DIPwithPyTorch\monolisa.png)

![water](E:\USTC\DIP\DIP-Teaching\Assignments\02_DIPwithPyTorch\water.png)

### 2. Pix2Pix implementation.



因为显卡限制所以批次改到了20.



## Training

run:

```bash
bash download_cityscapes_dataset.sh
python train.py
```



## Result

因为批次改到了20,而且数据集也换了稍大一些的cityscapes,所以下面只是迭代100次的结果示例,(loss因为关机了没保存,只有验证集图像)更多信息参考train_result和val_result

![result_1](Pix2Pix\val_results\epoch_100\result_1.png)

![result_2](Pix2Pix\val_results\epoch_100\result_2.png)

![result_3](Pix2Pix\val_results\epoch_100\result_3.png)

![result_4](Pix2Pix\val_results\epoch_100\result_4.png)

![result_5](Pix2Pix\val_results\epoch_100\result_5.png)
