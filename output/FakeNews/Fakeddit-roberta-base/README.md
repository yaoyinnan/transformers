## Fakeddit

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

stage_1:

stage_2: 顺序1W， 3epoch

    dev：f1-macro = 0.7824
    test: f1-macro = 0.7864

stage_3: 平衡2W， 1epoch

    dev：f1-macro = 0.8076
    test: f1-macro = 0.8096

stage_4: 平衡1W， 5epoch

    dev：f1-macro = 0.8033
    test: f1-macro = 0.8023

stage_5: 平衡4W， 1epoch

    dev：f1-macro = 0.8139
    test: f1-macro = 0.8141

stage_6: 平衡10W， 1epoch

    dev：f1-macro = 0.8300
    test: f1-macro = 0.8311
