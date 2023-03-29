import json
import ast

out_write=open("data/Belle_1M.train.json","a")#利用追加模式,参数从w替换为a即可
with open('data/belle_open_source_1M.train.json', 'r') as f:
    for line in f.readlines():##readlines(),函数把所有的行都读取进来；
        #print(json.loads(line)['input'])   
        # Input:\n{input} 
        out_dic ={}
        out_dic["input"] = "{input} ".format_map(json.loads(line))
        out_dic["target"] ="{target}".format_map(json.loads(line))
        out_write.write(str(out_dic)+'\n')



out_write.close()
