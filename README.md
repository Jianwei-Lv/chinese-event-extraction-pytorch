# chinese-event-extraction-pytorch
This is simple demo of chinese event extraction with pytorch


# 环境
python 3.6

torch==1.0.1

pytorch-pretrained-bert==0.6.2

# 数据集
数据集是从新闻网站上爬下来，人工标注的，不方便全部公开（虽然也没多少数据，而且标注质量也有待提高），nanhai_data文件夹下显示有10条json数据，以共参考数据格式。可以按照数据格式，更换自己的数据集。

# 效果
bert_RNN：
识别分类|P|R|F1
--|--|--|--
触发词识别|0.689|0.752|0.719
触发词分类|0.591|0.644|0.616
论元识别|0.547|0.702|0.615
论元分类|0.446|0.572|0.510

# 运行
运行 run.py

通过序列标注（BIO标签）同时识别分类触发词和实体，将识别分类的触发词特征和实体特征拼接，进行角色分类。
