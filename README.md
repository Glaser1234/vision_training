# vision_training

## 1. 环境依赖
- Ubuntu 22.04
- cmake version 3.22.1
- g++ 11.4.0
- Eigen3(libeigen3-dev)
- Ceres Solver(libceres-dev)
## 2. 构建与运行
``` bash
cmake -S . -B build
cmake --build build -j4
./build/task1
./build/task2
python3 plot_result.py
./build/task3 resources/task_3.mp4
./build/task3 resources/task_4.mp4
```
## 3. 输入素材
```
resources/
├── test_image.jpg     # 任务1 郁金香图片
├── task_2.mp4         # 任务2 合成旋转视频
├── task_3.mp4         # 任务3 小能量机关
└── task_4.mp4         # 任务3 大能量机关
```
## 4. 输出目录结构
```
result/
├── task1_images/
├── task2_fit/
├── task2_fit_result.md
└── task3_windmill/
```
## 5. 任务1：OpenCV 图片处理

**代码位置**：`src/task1_image/main.cpp`

### 5.1. 读图与颜色转换
**处理**：
- 使用 `imread("resources/test_image.jpg")` 读取原图
- 用 `cvtColor(img, gray, COLOR_BGR2GRAY)` 得到灰度图。
- 用 `cvtColor(img, hsv, COLOR_BGR2HSV)` 转换到 HSV，再用 `split` 拆出 H、S、V 三个单通道。

**结果图**：
- `result/task1_images/gray.png`
- `result/task1_images/hsv_h.png`
- `result/task1_images/hsv_s.png`
- `result/task1_images/hsv_v.png`

### 5.2. 滤波对比

**处理**：
对灰度图分别做均值、高斯、中值滤波。

**参数**：

| 滤波 | 函数 | 参数 |
|---|---|---|
| 均值 | `blur` | `Size(5,5)` |
| 高斯 | `GaussianBlur` | `Size(5,5), sigmaX=1.5` |
| 中值 | `medianBlur` | `ksize=5` |

**效果对比**：
- 均值滤波：邻域平均，平滑最强，花瓣边缘和细节最模糊。
- 高斯滤波：按高斯权重平均，平滑适中，边缘保留相对最好，适合作为常见预处理。
- 中值滤波：取邻域中位数，对椒盐噪声最好，边缘介于均值和高斯之间。

**结果图**：
- `result/task1_images/mean_filter.png`
- `result/task1_images/gaussian_filter.png`
- `result/task1_images/median_filter.png`

### 5.3. 红色提取
**处理**：
在 HSV 空间用 `inRange` 做双区间红色掩膜，再 `bitwise_or` 合并。

**阈值**：
- 低区间：`Scalar(0, 100, 100)` ~ `Scalar(10, 255, 255)`
- 高区间：`Scalar(170, 100, 100)` ~ `Scalar(179, 255, 255)`
- 合并：`bitwise_or(maskLow, maskHigh, mask)`

**红色、黄色边缘、阴影处理效果说明**：
- **红色**：红色在 HSV 中跨越 H 区间两端（0 附近和 179 附近），双区间合并后能完整覆盖红色花瓣主体，`red_mask.png` 里红色区域为白。
- **黄色边缘**：黄色 H 大约在 20–35，不在两个红色区间内，因此被排除，`red_mask.png` 里看不到黄色花瓣边缘。
- **阴影**：阴影区域 V 偏低，被 V>100 的下限排除，因此 `red_mask.png` 里不包含阴影区域。

**结论**：双区间阈值能覆盖红色花瓣，同时对黄色边缘和阴影有较好的区分能力。

**结果图**：
- `result/task1_images/red_mask.png`

### 5.4. 形态学与轮廓

**处理**：
1. 对红色掩膜分别做腐蚀、膨胀、开运算、闭运算，核为 `MORPH_RECT, Size(5,5)`。
2. 选择合适结果提外轮廓：用 `MORPH_OPEN` 后接 `MORPH_CLOSE` 串联得到 `contourSource`。
3. `findContours(contourSource, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`。
4. 按面积筛选：`contourArea(contours[i]) >= 500`。
5. 再按外接矩形长宽比筛选：`0.2 <= width/height <= 5.0`。
6. 在原图副本上绘制轮廓和外接矩形，并用 `putText` 把筛选后的轮廓面积写在外接矩形左上角。

**参数**：
- 形态学核：`MORPH_RECT, Size(5,5)`
- 面积阈值：`500.0`
- 长宽比范围：`[0.2, 5.0]`

**为什么选 `open→close`**：
- 开运算 `MORPH_OPEN`：先腐蚀后膨胀，去掉小白噪点；
- 闭运算 `MORPH_CLOSE`：先膨胀后腐蚀，填补小黑洞并连接窄断裂；
- 串联后轮廓比单独用 `open` 或 `closed` 更干净，噪声轮廓更少。

**轮廓面积**：
- 筛选后的轮廓面积已用 `putText` 标在 `contours_boxes.png` 上，红框左上角数字即为该轮廓面积（单位：像素²）。

**结果图**：
- `result/task1_images/erode.png`
- `result/task1_images/dilate.png`
- `result/task1_images/open.png`
- `result/task1_images/close.png`
- `result/task1_images/contours_boxes.png`（含轮廓面积）

### 5.5. 绘制与变换

**处理**：
1. 在原图副本上绘制圆、矩形和文字。
2. 绕图像中心旋转 35°。
3. 单独裁剪原图左上角 1/4。

**参数**：
- 圆：`circle(drawing, Point(200,200), 50, Scalar(255,0,0), 2)`
- 矩形：`rectangle(drawing, Rect(200,200,100,50), Scalar(0,255,0), 2)`
- 文字：`putText(drawing, "TASK1", Point(50,50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0,0,255), 2)`
- 旋转中心：`Point2f(img.cols/2.0, img.rows/2.0)`
- 旋转角度：`35°`
- 缩放：`1.0`
- 裁剪 ROI：`Rect(0, 0, img.cols/2, img.rows/2)`

**结果图**：
- `result/task1_images/drawing.png`
- `result/task1_images/rotated_35deg.png`
- `result/task1_images/crop_top_left.png`

### 任务1 结果图总索引（共 16 张）

| 文件名 | 对应步骤 |
|---|---|
| gray.png | 灰度图 |
| hsv_h.png | H 通道 |
| hsv_s.png | S 通道 |
| hsv_v.png | V 通道 |
| mean_filter.png | 均值滤波 |
| gaussian_filter.png | 高斯滤波 |
| median_filter.png | 中值滤波 |
| red_mask.png | 红色掩膜 |
| erode.png | 腐蚀 |
| dilate.png | 膨胀 |
| open.png | 开运算 |
| close.png | 闭运算 |
| contours_boxes.png | 轮廓与外接矩形（含面积） |
| drawing.png | 绘制圆、矩形、文字 |
| rotated_35deg.png | 旋转 35° |
| crop_top_left.png | 左上角 1/4 裁剪 |

# 任务2：合成旋转视频参数拟合

## 项目简介
本项目完成对合成旋转视频中目标点运动参数的提取与拟合。
输入视频为：`resources/task_2.mp4`
视频中白色点为旋转中心，青色点为运动目标。
已知旋转中心(480,360)，通过图像处理方法检测目标点位置，并根据目标点与旋转中心的相对位置计算角度变化.

##运行环境
- OpenCV
- Ceres Solver
- C++17
- Python3
- numpy
- pandas
- matplotlib


## 输出结果

程序运行后会在 `result/task2_fit/` 下生成结果文件：

| 文件 | 作用 |
|---|---|
| tracking_overlay.mp4 | 目标跟踪视频，显示检测到的目标点、旋转中心和角度信息 |
| data.csv | 保存每一帧检测得到的位置和展开后的角度数据 |
| fit.csv | 保存实际角度、拟合角度以及角度残差，用于绘制拟合效果图 |
| omega_fit.csv | 保存拟合得到的角速度曲线数据 |
| fit_comparison.png | 观测角度与拟合角度对比图 |
| angular_velocity.png | 实际角速度与拟合角速度对比图 |
| residuals.png | 角度拟合残差图 |
| task2_fit_result.md | 参数拟合结果报告，包括模型、参数和误差指标 |


**详细模型、方法、约束、初值与求解状态**：见 `result/task2_fit_result.md`。

## 7. 任务3：能量机关跟踪
**代码位置**：`src/task3_windmill/main.cpp`

### 7.1. 场景对应关系

| 输入视频 | 场景 | 同时亮起目标数 |
|---|---|---|
| `resources/task_3.mp4` | 小能量机关（1 个风车） | 最多 1 个 |
| `resources/task_4.mp4` | 大能量机关（2 个并排风车） | 最多 2 个 |

### 7.2. 实际视频参数

| 视频 | 分辨率 | 帧率 | 帧数 | 时长 |
|---|---|---|---|---|
| task_3.mp4 | 1440×1080 | 30 fps | 796 | 约 26.5 s |
| task_4.mp4 | 1440×1080 | 30 fps | 1800 | 60 s |

真实录像缺少可靠的逐帧采集时间，播放时间不能直接视为现场运动时间，因此本任务只考查视觉识别与稳定跟踪，不做时间拟合。

### 7.3. 识别内容

- 逐帧检测 **R 标中心**（圆环 + R 字实心块），并随相机运动持续跟踪（全局位移外推），角度一律以当前 R 标为参考计算；
- 检测亮起叶片的 **扇叶圆**（弧形灯串外端的圆形亮块）；
- 在画面上叠加：R 标中心、目标扇叶圆轮廓、扇叶圆中心、两中心连线、目标 ID 与跟踪状态（detected / lost）。

### 7.4. 运行

```bash
cmake -S . -B build
cmake --build build -j4
./build/task3 resources/task_3.mp4
./build/task3 resources/task_4.mp4
```

输出（与输入帧数、帧率、时序一致）：

- `result/task3_windmill/task_3/recognition_overlay.mp4`
- `result/task3_windmill/task_4/recognition_overlay.mp4`
- `result/task3_windmill/task_3/binary_process.mp4`（二值化中间过程）
- `result/task3_windmill/task_4/binary_process.mp4`

检测方法、锁定/重选规则与已知失败情况详见 `result/task3_tracking_result.md`。
