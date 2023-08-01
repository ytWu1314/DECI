[原论文：Graph Convolutional Networks for Event Causality Identification with Rich Document-level Structures](https://aclanthology.org/2021.naacl-main.273.pdf). 

[原链接](https://github.com/Bread27/DECI)


1. 创建虚拟环境:
```
conda create -n evcausality python=3.6 && conda activate evcausality
```

```python
# 我没有配置conda，使用的是
python -m venv my_venv
cd my_venv/Scripts
./Activate.ps1
```

2. 安装必要的库:
```
pip install -r requirements.txt

pip install torch==1.9.0+cu102 torchvision==0.10.0+cu102 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# 下载cc.en.300.bin
https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
```

遇到报错nltk缺少文件
```
# 方法1
import nltk
nltk.download('stopwords')
nltk.download('') #缺啥补啥
```

```
# 方法2
下载 https://github.com/nltk/nltk_data
将nltk_data-gh-pages中的package文件夹  -> C:\Users\用户名\AppData\Roaming\nltk_data
```


3. Preprocess data for EventStoryLine:
```
python prepare_data.py
```


4. Train the model:
```
python train.py
```


