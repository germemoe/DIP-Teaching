import cv2
import numpy as np
import gradio as gr
from scipy.interpolate import griddata

# 初始化全局变量，存储控制点和目标点
points_src = []
points_dst = []
image = None

# 上传图像时清空控制点和目标点
def upload_image(img):
    global image, points_src, points_dst
    points_src.clear()  # 清空控制点
    points_dst.clear()  # 清空目标点
    image = img
    return img

# 记录点击点事件，并标记点在图像上，同时在成对的点间画箭头
def record_points(evt: gr.SelectData):
    global points_src, points_dst, image
    x, y = evt.index[0], evt.index[1]  # 获取点击的坐标
    
    # 判断奇偶次来分别记录控制点和目标点
    if len(points_src) == len(points_dst):
        points_src.append([x, y])  # 奇数次点击为控制点
    else:
        points_dst.append([x, y])  # 偶数次点击为目标点
    
    # 在图像上标记点（蓝色：控制点，红色：目标点），并画箭头
    marked_image = image.copy()
    for pt in points_src:
        cv2.circle(marked_image, tuple(pt), 1, (255, 0, 0), -1)  # 蓝色表示控制点
    for pt in points_dst:
        cv2.circle(marked_image, tuple(pt), 1, (0, 0, 255), -1)  # 红色表示目标点
    
    # 画出箭头，表示从控制点到目标点的映射
    for i in range(min(len(points_src), len(points_dst))):
        cv2.arrowedLine(marked_image, tuple(points_src[i]), tuple(points_dst[i]), (0, 255, 0), 1)  # 绿色箭头表示映射
    
    return marked_image

# 执行仿射变换

def point_guided_deformation(image, source_pts, target_pts, alpha=1.0, eps=1e-8):
    """ 
    Return
    ------
        A deformed image.
    """
    
    warped_image = np.array(image)
    new_image = np.zeros_like(warped_image)

    height, width = warped_image.shape[:2]
    grid_size = 10

    x_points = np.arange(0, width, grid_size)
    y_points = np.arange(0, height, grid_size)

    if width % grid_size != 0:
        x_points = np.append(x_points, width - 1)
    if height % grid_size != 0:
        y_points = np.append(y_points, height - 1)

    grid_x, grid_y = np.meshgrid(x_points, y_points)
    new_grid_x = np.zeros_like(grid_x)
    new_grid_y = np.zeros_like(grid_y)
    A = np.zeros(source_pts.shape[0])
    source_pts, target_pts = target_pts, source_pts
 
    for i in range(grid_x.shape[0]):
        for j in range(grid_y.shape[1]):
            v = np.array([grid_x[i, j], grid_y[i, j]])
            new_v = np.zeros(2)
           
            w = 1.0 / (eps + np.power(np.linalg.norm(source_pts - v, axis=1), (2. * alpha)))
            p_star = np.sum(np.transpose([w, w]) * source_pts, axis=0) / np.sum(w)
            q_star = np.sum(np.transpose([w, w]) * target_pts, axis=0) / np.sum(w)
            p_hat = source_pts - p_star
            q_hat = target_pts - q_star
            temp_matrix_inv = np.zeros((2,2))
            for k in range(source_pts.shape[0]):
                temp_matrix_inv += w[k] * np.outer(p_hat[k], p_hat[k])
            temp_matrix_inv = np.linalg.inv(temp_matrix_inv + np.eye(2)*eps)
            for k in range(source_pts.shape[0]):
                A[k] = w[k] * (v - p_star).dot(temp_matrix_inv).dot(p_hat[k])
                new_v += A[k] * q_hat[k]
            new_v += q_star
            new_grid_x[i,j], new_grid_y[i,j] = new_v
    full_grid_x, full_grid_y = np.meshgrid(np.arange(0, width), np.arange(0, height))


    new_full_x = griddata((grid_x.ravel(), grid_y.ravel()), new_grid_x.ravel(),
                        (full_grid_x, full_grid_y), method='linear')

    new_full_y = griddata((grid_x.ravel(), grid_y.ravel()), new_grid_y.ravel(),
                        (full_grid_x, full_grid_y), method='linear')
    map_x = np.clip(np.floor(new_full_x).astype(int), 0, width - 1)
    map_y = np.clip(np.floor(new_full_y).astype(int), 0, height - 1)
    for i in range(height):
        for j in range(width):
            new_image[i, j] = warped_image[map_y[i, j], map_x[i, j]]

    return new_image

def run_warping():
    global points_src, points_dst, image ### fetch global variables

    warped_image = point_guided_deformation(image, np.array(points_src), np.array(points_dst))

    return warped_image

# 清除选中点
def clear_points():
    global points_src, points_dst
    points_src.clear()
    points_dst.clear()
    return image  # 返回未标记的原图

# 使用 Gradio 构建界面
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", label="上传图片", interactive=True, width=800, height=200)
            point_select = gr.Image(label="点击选择控制点和目标点", interactive=True, width=800, height=800)
            
        with gr.Column():
            result_image = gr.Image(label="变换结果", width=800, height=400)
    
    # 按钮
    run_button = gr.Button("Run Warping")
    clear_button = gr.Button("Clear Points")  # 添加清除按钮
    
    # 上传图像的交互
    input_image.upload(upload_image, input_image, point_select)
    # 选择点的交互，点选后刷新图像
    point_select.select(record_points, None, point_select)
    # 点击运行 warping 按钮，计算并显示变换后的图像
    run_button.click(run_warping, None, result_image)
    # 点击清除按钮，清空所有已选择的点
    clear_button.click(clear_points, None, point_select)
    
# 启动 Gradio 应用
demo.launch()
