## Fakeddit-3way

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

stage_1: 平衡51396， 1epoch

    dev：f1-macro = 0.7494
    test: f1-macro = 0.7465

stage_2: 41434 + 17132 + 41434 = 10W， 1epoch

    dev：f1-macro = 0.8022
    test: f1-macro = 0.8094
    
stage_3: 全部数据， 1epoch

    dev：f1-macro = 0.8576
    test: f1-macro = 0.8657 
       
stage_4: 全部数据， 10epoch

    dev：f1-macro = 0.8733
    test: f1-macro = 0.8716

