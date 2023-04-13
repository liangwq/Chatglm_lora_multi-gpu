## 数据可以到huggingface数据集去下载 ##
## 地址：https://huggingface.co/datasets/BelleGroup/train_1M_CN/tree/main ##
给两份belle中文的self instruct数据
数据格式如下：
{'input': '将以下段落转化为三个句子的摘要。\\n\n该国领导人宣布将采取措施，以扭转经济下滑的趋势。这些措施包括减税、增加基础设施投资和提高消费者信心。领导人表示，这些措施将帮助提振经济，并增加就业机会。\\n ', 'target': '该国领导人宣布将采取措施扭转经济下滑的趋势。这些措施包括减税、增加基础设施投资和提高消费者信心。措施将有助于提振经济并增加就业机会。'}
{'input': '从给定的文本集合中找出两个语义最相似的句子。\n句子1：我今天很开心。 句子2：我今天心情不好。 句子3：今天是个好天气。 句子4：我喜欢去海滩。 句子5：我喜欢阅读。 ', 'target': '句子1和句子2是语义上最相似的，都是在描述今天的情绪状态。'}

1.0.5M版本：
wget https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN/resolve/main/Belle.train.json
2.1M版本
wget https://huggingface.co/datasets/BelleGroup/generated_train_1M_CN/resolve/main/belle_open_source_1M.train.json
3.把两份数据合并成一份
a.0.5M和1M数据字段有些不同，统一处理数据，用地下代码处理1M数据
python process_belle_1M_data.py
b.把两份文件合并成一份，命名为：Belle_0_1.train.json
cat Belle.train.json Belle_1M.train.json>Belle_0_1.train.json
