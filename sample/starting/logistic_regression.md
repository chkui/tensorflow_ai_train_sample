<<<<<<< HEAD
# 逻辑回归（logistics ）

标签（空格分隔）： 未分类

---
标准的直线性结构很难完全满足多种多样的样本分布，尤其是对结论进行分类的问题。因此一些更加特殊结构用于ML的分类问题。logistics回归是在线性结果的基础上使用sigmoid函数对结论进行一次(0,1)分布处理，他的最大的特点是将(-∞, +∞)的线性数据规约到(0,1)之间。其结构为：
$f(x) = \frac{1}{1+e^{-x}}$

## 建模

设定(1):现在有$m$个训练样本以及标签、共有$n$个特征，得到权重方程：

$g_i=g(x)_i=w_{j}x_{ij} ,其中i∈(0,m],j∈[0,n)$,n可以取0值表示$x_0=1$
设定(2):公式中$g_i$表示权重公式的计算结果，$y_i$表示每一个样本的标签（真实结果）。
因此我们需要找到$w_j$的权重数据使得$g_i$无限去接近$y_i$。

由于样本的标签只取值0或1，所以可以用sigmoid函数创建一个2分问题的预测模型：

设$Y_w$表示预测模型$g_i$根据条件参数$w_j$取值为1的概率。
$那么使用sigmoid函数得到一个概率表达式：Y_w = \frac{1}{1 + e^{-g_i}}$
$则g_i = 0的概率为 1 - Y_w = 1 - \frac{1}{1+e^{-g_i}}$
$=>1 - Y_w = \frac{1+e^{-g_i}}{1+e^{-g_i}} - \frac{1}{1+e^{-g_i}}$
$=\frac{e^{-g_i}}{1+e^{-g_i}}，对其换底（e^{-g}=\frac{1}{e^g}）$
得到：
$1 - Y_w = \frac{1}{1+e^{g_i}}$

用条件概率表达以上2个结论：
$p(y_i=1|x_{ij})=\frac{1}{1+e^{-g_i}}，条件x_{ij}导致y_i为1的概率$
$p(y_i=0|x_{ij})=\frac{1}{1+e^{g_i}}，条件x_{ij}导致y_i为0的概率$
且，$g_i = w_{ij}x_j$

有了以上条件概率，可以使用**极大似然估计法**来根据样本（数量为$m$）获得$w_{j}$的取值。

因为任意$y_i$的取值为0或者1，根据这个性质可以得到单个样本$k$的概率公式为：
$(Y_w)^{y_k}(1-Y_w)^{y_k}$
因此，整个样本空间对$w_j$的**极大似然估计**公式为：
$L(w_j) = \prod_{i=1}^m{(Y_w)^{y_i}(1-Y_w)^{1-y_i}}$
使用对数简化计算：
$L_W = \ln[L(w_j)]=\sum_{i=1}^{m}[y_i\ln(Y_w) + (1-y_i)\ln(1-Y_w)]$
如果使用梯度递减，考察似然函数概率接近最大的情况，那么：
$L_{W_j}=\frac{1}{m}L_W=\frac{1}{m}\sum_{i=1}^{m}[y_i\ln(Y_w) + (1-y_i)\ln(1-Y_w)]$

公式$L_{w_j}$就是评估函数，他是一个概率公式，理论上无线接近于1。
$Y_w=\frac{1}{1+e^{-w_{j}x_j}}$最终用于评估结果为1的概率。
对应的：
$1-Y_w=\frac{1}{1+e^{g_i}}$是结果0的概率。
当输入一个新的独立样本$X^{(k)}（特征:x_{1}^{(k)},x_{2}^{(k)},x_{3}^{(k)},…x_{n}^{(k)}）$时，可以利用这个模型计算结果被分类为那边的。

## 回归训练
有了模型和LOSS函数，接下来就是计算$w_j$的取值，用梯度递减，对评估函数求偏导函数：$\frac{∂L_{w_j}}{∂w_j}$。
考察只有一个样本的情况（$m=1$），将下标i替换为1分别对$w_j$求偏导 :
$\frac{∂L_{w_j}}{∂w_j}=\frac{y_1}{Y_w}(Y_w)^{'} - \frac{1-y_1}{1-Y_w}(Y_w)^{'}$
$\frac{∂L_{w_j}}{∂w_j}=(\frac{y_1}{Y_w}-\frac{1-y_1}{1-Y_w})(Y_w)^{'}$

---

**继续求解这个方程组可以使用以下数学特性**
接下来利用一些数学性质来继续求导。
事件为1和0的概率比为：
$odds = \frac{p(y_i=1|x_{ij})}{p(y_i=0|x_{ij})} = \frac{Y_w}{1-Y_w}$
展开得到：
$odds=\frac{\frac{1}{1+e^{-g(x)}}}{\frac{1}{1+e^{g(x)}}}=\frac{\frac{1}{1+e^{-g(x)}}}{\frac{e^{-g(x)}}{1+e^{-g(x)}}}=e^{g(x)}$
所以：
$\ln(odds) = \ln \frac{Y_w}{1-Y_w} = g(x) = w_{j}x_j = w_0x_0 + w_1x_1 + w_2x_2 + ... + w_jx_j, (x_0=1)$

---

继续求$w_j$的偏导函数：
$\frac{∂L_{w_j}}{∂w_j}=(\frac{y_1}{Y_w}-\frac{1-y_1}{1-Y_w})(\frac{1}{1+e^{-g(x)}})^{'}$
$=(\frac{y_1}{Y_w}-\frac{1-y_1}{1-Y_w})(\frac{e^{g(x)}}{1+e^{g(x)}})^{'}$
$=(\frac{y_1}{Y_w}-\frac{1-y_1}{1-Y_w})\frac{e'^{g(x)}(1+e^{g(x)})-e^{g(x)}(1+e^{g(x)})^{'}}{(1+e^{g(x)})^2}$
$=(\frac{y_1}{Y_w}-\frac{1-y_1}{1-Y_w})\frac{e'^{g(x)}+e'^{g(x)}e^{g(x)})-e^{g(x)}e'^{g(x)}}{(1+e^{g(x)})^2}$
$=(\frac{y_1}{Y_w}-\frac{1-y_1}{1-Y_w})\frac{e'^{g(x)}}{(1+e^{g(x)})^2}$
$=(\frac{y_1}{Y_w}-\frac{1-y_1}{1-Y_w})e^{g(x)}\frac{1}{(1+e^{g(x)})^2}g'(x)$
带入$e^{g(x)}=\frac{Y_w}{1-Y_w}$:
$=(\frac{y_1}{Y_w}-\frac{1-y_1}{1-Y_w})\frac{Y_w}{1-Y_w}\frac{1}{(1+\frac{Y_w}{1-Y_w})^2}g'(x)$
$=(\frac{y_1}{1-Y_w}-\frac{Y_w-y_1Y_w}{(1-Y_w)^2})\frac{1}{(\frac{1}{1-Y_w})^2}g'(x)$
$=\frac{y_1-y_1Y_w-Y_w+y_1Y_w}{(1-Y_w)^2}(1-Y_w)^2g'(x)$
$=(y_1-Y_w)g'(x)$
准确的表达式为：
$\frac{∂L_{w_j}}{∂w_j}=(y_1-Y_w)\frac{∂}{∂w_j}G(w_j,x_{1j})$
$\frac{∂L_{w_j}}{∂w_j}=(y_1-Y_w)\frac{∂}{∂w_j}(w_0+w_1x_{11}+w_2x_{12}+...+w_1x_{1j})$
所以对任意w_k求偏导数的结果为：
$\frac{∂L_{w_j}}{∂w_j}=(y_1-Y_w)x_{1j}$
$\frac{∂L_{w_j}}{∂w_j}=(y_1-\frac{1}{1+e^{-g(x)}})x_{1k}$
$\frac{∂L_{w_j}}{∂w_j}=(y_1-\frac{1}{1+e^{-\sum_{j=0}^{n} w_jx_{1j}}})x_{1k}，x_{i0}=1$
扩展到整个样本空间：
$\frac{∂L_{w_j}}{∂w_j}=\frac{1}{m}\sum_{i=1}^m(y_1-\frac{1}{1+e^{-\sum_{j=0}^{n} w_jx_{1j}}})x_{1k}，x_{i0}=1$

## 总结
$\eta$表示步长，且令$\nabla_j=\frac{∂L_{w_j}}{∂w_j}=\frac{1}{m}\sum_{i=1}^m(y_i-\frac{1}{1+e^{-\sum_{j=0}^{n} w_jx_{1j}}})x_{ik}$。
那么，梯度递减的更新公式为：
$w_j = w_j - \eta\nabla_j$
预测单个样本偏向1的概率：
$P(y=1|x_j)=(1+e^{-\sum_{j=0}^{n}w_jx_j})^{-1}$
预测单个样本偏向0的概率：
$P(y=0|x_j)=(1+e^{\sum_{j=0}^{n}w_jx_j})^{-1}$
回归评估函数：
$L(w_{ij},x_{ij},y_i)=\frac{1}{m}\sum_{i=1}^{m}[y_i\ln(Y_w) + (1-y_i)\ln(1-Y_w)]$
$=\frac{1}{m}\sum_{i=1}^{m}[-y_i\ln(1+e^{-\sum_{j=0}^nw_{ij}x_{ij}}) + (y_i-1)\ln(1+e^{\sum_{j=0}^nw_{ij}x_{ij}})]$

## 矩阵运算
设$m=3,n=3$，则特征的矩阵结构为,这里的：
权重矩阵：
$W_j=\begin{bmatrix}w_0&w_1&w_2&w_3\end{bmatrix}$

特征矩阵：
$X=X_{ij}=\begin{bmatrix}x_{10}&x_{11}&x_{12}&x_{13}\\x_{20}&x_{21}&x_{22}&x_{23}\\x_{30}&x_{31}&x_{32}&x_{33}\end{bmatrix}$
其中$x_{i0}$恒等于1。

标签标签矩阵：
$Y=Y_i=\begin{bmatrix}y_1\\y_2\\y_3\end{bmatrix}$
$y_i取值[0,1]$

设$G=G_i=X_{ij}×W_j^T$,其结构为：
$G=\begin{bmatrix}g_1\\g_2\\g_3\end{bmatrix}$

### 算法
#### 1.矩阵方法求$e^{w_jx_{ij}}$
$令E_i = e^{-\sum_{j=0}^nw_{ij}x_{ij}}$。相当于当行求线性权重值，然后再计算exp。
$E_1=e^{-g_1}$
$E_2=e^{-g_2}$
$E_3=e^{-g_3}$
所以，
$Ei=exp(-G)=exp(-X×W^T)$
所以当$m=3,n=3$时：
$矩阵Ei=exp(\begin{bmatrix}-(w_0+w_1x_{11}+w_2x_{12}+w_3x_{13})\\-(w_0+w_1x_{21}+w_2x_{22}+w_3x_{23})\\-(w_0+w_1x_{31}+w_2x_{32}+w_3x_{33})\end{bmatrix})$
所以：
$E_i^{-}=e^{\sum_{j=0}^nw_{ij}x_{ij}}=exp(X×W^T)$
$exp()$表示对矩阵中每一个元素求e的指数。$E_i$和$E_i^-$的形状为(m,1)
#### 2.按矩阵中的元素计算$\ln和加减:$
$y_i\ln(1+E_i)=\begin{bmatrix}y_1&y_2&y_3\end{bmatrix}×\begin{bmatrix}ln(1+exp(-g_1))\\ln(1+exp(-g_2))\\ln(1+exp(-g_3))\end{bmatrix}$

$=Y_i×ln(1+exp(-X×W^T))$

$ln()$表示对矩阵中每一个元素求e的对数。

所以：

$(1-y_i)ln(1+E_i^{-})=\begin{bmatrix}(1-y1)ln(1+(w_0+w_1x_{11}+w_2x_{12}+w_3x_{13}))\\(1-y2)ln(1+(w_0+w_1x_{21}+w_2x_{22}+w_3x_{23}))\\(1-y3)ln(1+(w_0+w_1x_{31}+w_2x_{32}+w_3x_{33}))\end{bmatrix}$
$=(1-Y_i^T)×ln(1+exp(X×W^T))$
所以合并公式，评估函数的矩阵结构为：
$L=Y_i×ln(1+exp(-W×X^T))+(1-Y_i)×ln(1+exp(W×X^T))$

L是一个形状为(m, 1)的矩阵，对所有的$w_{ij}$进行评估则对举证的所有元素求和再除以m

####3.梯度递减
$\nabla_j=\frac{∂L_{w_j}}{∂w_j}=\frac{1}{m}\sum_{i=1}^m(y_i-\frac{1}{1+e^{-\sum_{j=0}^{n} w_jx_{1j}}})x_{ik}$
$\nabla=X×(Y^T-(1+E)^I)/m$
$\nabla=X×(Y^T-(1+exp(-X×W^T))^I)/m$
**更新权重矩阵**
$W = W-\eta\nabla$
=======
		<math>
			<mrow>
				<mi>A</mi>
				<mo>=</mo>
				
				<mfenced open="[" close="]">
					<mtable>
						<mtr>
							<mtd><mi>x</mi></mtd>
							<mtd><mi>y</mi></mtd>
						</mtr>
						
						<mtr>
							<mtd><mi>z</mi></mtd>
							<mtd><mi>w</mi></mtd>
						</mtr>
					</mtable>
				</mfenced>
			</mrow>
		</math>
>>>>>>> 15f31a76d7498fd61b62a74e986c251a696b8e82
