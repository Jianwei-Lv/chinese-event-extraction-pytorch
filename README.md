# chinese-event-extraction-pytorch
一个简单的用pytorch实现中文事件抽取的代码，写得不好，有待提高，希望对大家有所帮助。有问题欢迎留言。

# 环境
python 3.6

torch==1.0.1

pytorch-pretrained-bert==0.6.2

# 数据集
数据集是从新闻网站上爬下来，人工标注的，不方便全部公开（虽然也没多少数据，而且标注质量也有待提高），nanhai_data文件夹下显示有10条json数据，以共参考数据格式。可以按照数据格式，更换自己的数据集。贡献人员：Lei Li; Panpan Jin; Kaiwen Wei; Jianwei Lv; Xiaoyu Li。

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

# 预训练语言模型
bert模型放在bert_pretain目录下，三个文件：

pytorch_model.bin

bert_config.json

vocab.txt

预训练模型下载地址：

bert_Chinese: 模型 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz

词表 https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-vocab.txt

来自[这里](https://github.com/huggingface/pytorch-transformers)

# 参考仓库

项目1：https://github.com/649453932/Bert-Chinese-Text-Classification-Pytorch

项目2：https://github.com/nlpcl-lab/bert-event-extraction
