# 载入模块
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
a = np.arange(-20.20,1)
b = 1.5*a + 3
z = np.log(b)

# 创建 3D 图形对象
fig = plt.figure()
plt.xlabel('a')
ax = Axes3D(fig)
ax.set_xlabel('a Label')
ax.set_xlabel('a Label')
ax.set_xlabel('a Label')

# 绘制线型图
ax.plot(a, b, z)

# 显示图
plt.show()