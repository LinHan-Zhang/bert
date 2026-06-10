# 情感词表来源

本目录词表由 **大连理工大学情感词汇本体（DLUT）** 与 **训练集挖掘词** 共同构建。

- 原始资源：[yizhanmiao/DLUT-Emotionontology](https://github.com/yizhanmiao/DLUT-Emotionontology)
- 本地原始文件：`raw/dlut.csv`
- 融合词表：`pos_merged.txt` / `neg_merged.txt`（DLUT 筛选 + train 挖掘 + 默认词）
- 引用声明：使用了大连理工大学信息检索研究室的情感词汇本体
- 参考文献：徐琳宏, 林鸿飞, 潘宇, 等. 情感词汇本体的构造[J]. 情报学报, 2008, 27(2): 180-185.

构建脚本：`src/build_lexicon.py`

```bash
cd /root/autodl-tmp/src
python build_lexicon.py \
  --top-k 500 --min-hits 3 \
  --mine-top-k 250 --mine-min-hits 5 \
  --pos-file pos_merged.txt --neg-file neg_merged.txt
```
