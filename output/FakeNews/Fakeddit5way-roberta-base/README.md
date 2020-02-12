## Fakeddit-5way

### classification
2分类：

1. 真
2. 假

3分类：

1. 真
2. 带有真实文本的虚假新闻
3. 假

5分类：

1. 真
2. 讽刺或模仿（说反话）
3. 误导性内容（虚假内容）
4. 冒名顶替内容（机器生成）
5. 错误链接（图文不符）

### model

roberta-base

### results

stage_1: 44462 + 44462 + 44462 + 25058 + 27993 = 186437, 1epoch

    dev：f1-macro = 0.7412
    test: f1-macro = 0.7460

stage_2: 25058 + 25058 + 25058 + 25058 + 25058 = 125290(balance), 1epoch

    dev：f1-macro = 0.7087
    test: f1-macro = 0.7156

stage_3: 全部数据, 1epoch

    dev：f1-macro = 0.7988
    test: f1-macro = 0.8021

