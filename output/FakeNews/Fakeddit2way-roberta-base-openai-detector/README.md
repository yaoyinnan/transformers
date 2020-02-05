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

roberta-base-openai-detector

### results

stage_1: 平衡49W， 1epoch

    dev：f1-macro = 0.8556
    test: f1-macro = 0.8552


