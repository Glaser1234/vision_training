# 任务二：合成旋转视频的参数拟合结果


## 1. 拟合模型

根据任务要求，角速度满足：

$$
\omega(t)=b+A\sin(\Omega t+\phi)
$$


对角速度积分得到角度模型：

$$
\theta(t)=
\theta_0+bt+
\frac{A}{\Omega}
(\cos\phi-\cos(\Omega t+\phi))
$$


其中：

- A：角速度变化幅值，单位 rad/s；
- b：平均角速度，单位 rad/s；
- Ω：角速度变化频率参数，单位 rad/s；
- φ：相位，单位 rad；
- θ0：初始角度，单位 rad。


根据任务要求，拟合参数为：

$$
A,b,\Omega,\phi
$$


初始角度：

$$
\theta_0=\theta(0)
$$

直接由第一帧角度确定。


## 2. 目标检测与角度计算方法


首先读取视频帧，通过 HSV 阈值分割提取青色目标区域。


对检测到的目标区域计算质心：

$$
(x_i,y_i)
$$


根据已知旋转中心：

$$
(c_x,c_y)=(480,360)
$$


计算目标角度：

$$
\theta_i=
atan2(c_y-y_i,x_i-c_x)
$$


由于 atan2 输出范围为：

$$
[-\pi,\pi]
$$


因此对角度序列进行 unwrap 处理，消除旋转过程中由于周期变化产生的角度跳变，得到连续角度数据。


最终得到：

- 有效检测点数量：1440
- 时间范围：

$$
0\sim23.9833s
$$


## 3. 参数初始化方法


为了提高非线性优化收敛速度，对参数进行初始化。


### 初始角度 θ0


取第一帧展开后的角度：

$$
\theta_0=\theta(0)
$$


程序：

```cpp
theta0=data.front().theta_unwrapped;
```


### 平均角速度 b


利用整个运动过程估计：

$$
b=
\frac{\theta_{end}-\theta_0}
{t_{end}-t_0}
$$


程序：

```cpp
b=
(data.back().theta_unwrapped-data.front().theta_unwrapped)
/
(data.back().t-data.front().t);
```


### 角速度变化幅值 A


首先计算离散角速度：

$$
\omega_i=
\frac{\theta_i-\theta_{i-1}}
{t_i-t_{i-1}}
$$


根据最大值和最小值估计：

$$
A=
\frac{\omega_{max}-\omega_{min}}{2}
$$



### 频率参数 Ω


根据视频中速度变化周期进行初始化：

$$
\Omega=
\frac{2\pi}{4}
$$


即假设角速度变化周期约为 4 秒。



### 相位 φ


初始设置：

$$
\phi=0
$$



## 4. 优化方法与约束条件


采用 Ceres Solver 对角度模型进行非线性最小二乘优化。


优化目标：

$$
\min
\sum_i
(\theta_i-\theta(t_i))^2
$$


其中：

- θi 为视频检测得到的展开角度；
- θ(ti) 为模型计算角度。


求解器设置：

- 使用 AutoDiff 自动求导；
- 使用 Levenberg-Marquardt 优化方法；
- 使用 DENSE_QR 线性求解器；
- 最大迭代次数设置为 200。


### 参数约束


根据任务要求：

$$
A>0
$$


$$
b>A
$$


$$
\Omega>0
$$


程序中加入：

$$
A>0
$$


以及：

$$
\Omega>0
$$


约束。


同时对最终结果进行检查：

$$
b>A
$$


本次拟合结果满足该条件。



## 5. 求解状态


Ceres Solver 优化结果：

```
Iterations: 10

Initial cost:
6.264722e+01

Final cost:
3.898406e-03

Termination:
CONVERGENCE
```


优化过程正常收敛。



## 6. 最终拟合参数


得到参数：


$$
A=0.54997\ rad/s
$$


$$
b=1.35002\ rad/s
$$


$$
\Omega=1.64987\ rad/s
$$


$$
\phi=0.702961\ rad
$$


$$
\theta_0=0.350352\ rad
$$


参数满足：

$$
b>A
$$


即：

$$
1.35002>0.54997
$$


符合持续正方向旋转约束。



## 7. 误差指标


### 角度拟合误差


计算角度拟合 RMSE：

$$
RMSE_\theta
=
\sqrt{
\frac1N
\sum_i
(\theta_i-\theta(t_i))^2
}
$$


结果：

```
RMSEθ = 0.0023269 rad
```


### 角速度拟合误差


角速度拟合 RMSE：

```
RMSEω = 0.0595592 rad/s
```


## 8. 结果分析


角度拟合曲线与实际检测角度基本重合。


残差曲线在较小范围内波动，没有出现明显的角度展开错误或拟合发散。


角速度拟合结果能够反映角速度随时间的周期变化趋势。


整体结果表明，采用该模型能够较好描述视频中的旋转运动。
