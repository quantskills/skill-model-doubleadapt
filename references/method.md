# 方法说明

DoubleAdapt 将基础预测网络与两个适配器结合：

- Feature Adapter 根据输入特征与多个原型的相似度生成门控权重，再叠加多个特征变换；
- Label Adapter 根据输入特征生成门控权重，对成熟标签执行尺度和偏移变换；
- Support 数据用于计算适配方向；
- Query 数据用于评价适配后的泛化表现；
- 推理阶段只使用特征和冻结 checkpoint。
