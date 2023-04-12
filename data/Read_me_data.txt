## 数据可以到huggingface数据集去下载 ##
## 地址：https://huggingface.co/datasets/BelleGroup/train_1M_CN/tree/main ##
给两份belle中文的self instruct数据
1.0.5M版本：
wget https://huggingface.co/datasets/BelleGroup/generated_train_0.5M_CN/resolve/main/Belle.train.json
2.1M版本
wget https://huggingface.co/datasets/BelleGroup/generated_train_1M_CN/resolve/main/belle_open_source_1M.train.json
3.把两份数据合并成一份
a.0.5M和1M数据字段有些不同，统一处理数据，用地下代码处理1M数据
python process_belle_1M_data.py
b.把两份文件合并成一份，命名为：Belle_0_1.train.json
cat Belle.train.json Belle_1M.train.json>Belle_0_1.train.json
