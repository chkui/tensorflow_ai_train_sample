# （1）

## 什么是机器学习
### 1. 如何描述世界

人类总是通过自己所能理解的语言来理解一个事物，比如当一个小孩牙牙学语的时候，对于一辆车他只能描述车是大车还是小车，是红色还是是蓝色。

| 大小 | 颜色 |
|------|------|
|  大  |  红  |

到了成年一个人的后，认知水平和语言能力能够让人类可以描述更加复杂的内容，例如一辆车的变速箱是什么类型、排量大小等。

| 大小 | 颜色 | 变速箱 | 排量 |
|------|------|--------|------|
|  大  |  红  |  CVT   | 2.0T |

随着人们认知能力的增加这份内容还会不断的增多。想象一下我们在各种网站上看到的关于汽车的说明，几乎介绍每一辆车的格式都是一样的，比如排量多大、轴距多长、轮胎尺寸等等，无论是什么车都可以像填空题一样将不同的内容填入一个固定格式的表单中：

跳出汽车的例子，我们到网上的去买衣服，对一件的衣服的描述也是像做填空题一样向一个固定格式的表单添加数据——衣服是红色还是蓝色、是中码还是大码、是波西米亚风还是日韩风、是涤纶还是纯棉：

再看看家具、3C，他们都可以事先设计一份固定的表单，然后向表单中固定的位置填入不同的信息以表示同类产品之间的差异。

以此类推，户籍信息、学生成绩、个人简历、体检报告、国家的经济统计数据、城市天气、人的样貌等等通都可以事先设计好一个固定格式的表单，对同类事物的描述就是向表单中的空格填入对应的信息。

这些表单当然不会永远固定格式，就像小孩的认知到成年人的认知成长一样，随着人类社会的发展表单也在不断的革新。比如个人简历，三十年前没有手机一栏、二十年前没有Email一栏，而现在这已是人类的主要联系方式。

### 2.一些基本概念

前一小节反复用到**表单**的这个词汇，几乎所有的独立事物都可以通过一份**表单**来描述。比如描述一个人的相貌可以分为眼睛大或小、鼻梁高或塌、嘴唇厚或薄、脸型方或圆，根据这些内容针对一个人外貌的介绍就有了一份**表单**：

| 眼睛 | 嘴巴 | 鼻子 | 脸型 |
|:---:|:----:|:---:|:----:|
| 大 | 厚 | 高 | 方 |

再比如通过温度、风力、湿度、晴雨云雾来描述一个城市的天气情况：

| 城市 | 温度 | 风力 | 气象 |
|:---:|:---|:---|:---|
| 北京 | 25 | 6级 | 晴 |

这些表单中的描述的对象我们称之为**主题**。比如第一份表单描述的对象是“样貌”，所以他的主题就是样貌，而表单中的关于眼睛、鼻子的描述称之为 **特征**，比如“大”或“小”是眼睛的特征、“高”或“塌”是鼻子的特征。

人们通常能够根据特征得到一个结论，我们称之为**标签**。比如根据北京的天气描述，我们的结论是5月20号这天是“好天气”：

| 城市 | 温度 | 风力 | 气象 | 日期 | 结论 |
|:---:|:---|:---|:---|:---|:---|
| 北京 | 25 | 6级 | 晴 | 2018.5.20 | 好天气|

可以将相同主题的多分表单整合在一起形成一份清单，所以一个**主题**内可以有0～n条数据：

| 城市 | 温度 | 风力 | 气象 |
|:---:|:---|:---|:---|
| 上海 | 20 | 8级 | 大暴雨 |
| 广州 | 27 | 5级 | 多云 |
| 深圳 | 26 | 5级 | 多云 |

请记住**表单**、**主题**、**特征**、**标签**这些字眼及其所代表的含义，在后续的内容中会反复的出现。实际上这些概念都来源于ER数据库<sup>[1]</sup>或数据仓库（数据分析）<sup>[2]</sup>，但是没有这方面的知识背景也不妨碍理解接下来的内容，因为我们使用的都是日常生活的例子。

下面创建一个，本章后续将通过这个例子引入基本的人工智能训练的方法。

下面主题是人的外貌

| 姓名 | 眼睛 | 嘴巴 | 鼻子 | 脸型 | 样貌 |
|:---:|:---:|:----:|:---:|:----:|:---|
| Alice | 小 | 薄 | 塌 | 圆 | =婉约 |
| Bob | 大 | 厚 | 高 | 方 | =大气 |
| Claire | 大 | 薄 | 高 | 圆 | =婉约 |
| Deek | 大 | 厚 | 高 | 方 | =大气 |
| Echo | 大 | 薄 | 高 | 方 | =大气 |
| Frank | 大 | 薄 | 塌 | 方 | =大气 |
| Grace | 大 | 厚 | 高 | 圆 | =大气 |

整个表格就是一份**表单**,里面的内容Alice、小、薄

在以上的清单中作者根据个人的看法将人的样貌分为2类——大气、婉约。
笔者的分类原则很简单，就是按照以下几点一步一步的进行判断：
1. 凡是嘴厚脸型方的都归类为大气。而嘴薄脸型圆的归类为婉约。
2. 如果嘴薄脸方或嘴大脸小则考察眼睛是否大，大=大气。
3. 如果眼睛小则考则鼻子， 高则大气，塌则婉约。

按照以上这三点分类原则，向清单中加入任何人的特征我们都可以马上得到一个结论：

|  | 眼睛 | 嘴巴 | 鼻子 | 脸型 | 样貌 |
|:---:|:---:|:----:|:---:|:----:|:---|
| Kale | 小 | 厚 | 高 | 圆 | =大气 |
| Henry | 小 | 厚 | 塌 | 圆 | =婉约 |

其实上面的1、2、3点就是一个算法，有编码能力的人都可以根据这个算法得到生成任意大小的清单，清单中的数据都符合这个规则。

下面的代码随机生成了200行数据的清单，清单中所有的数据都符合

---

[1] 对于ER数据库来说，主题就是面向实体设计的一系列ER表，它是以一种反规范化（denormalization）的方式呈现的，而在标准的ER关系中往往会为城市单独创建一张字典表。
[2]表单、主题、特征的概念与数据仓库中的概念几乎一致，在使用海量数据进行AI训练之前要做的工作是处理海量的数据，这一部分工作往往交给数据分析和大数据处理。