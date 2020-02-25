## Fakeddit-FienGrained

### classification
细粒度分类（21类）：

nottheonion, 
fakealbumcovers, 
confusing_perspective, 
pareidolia, 
upliftingnews, 
pic, 
mildlyinteresting, 
fakehistoryporn, 
theonion, 
photoshopbattles, 
misleadingthumbnails, 
usnews, 
propagandaposters, 
subredditsimulator, 
usanews, 
neutralnews, 
satire, 
savedyouaclick, 
subsimulator gpt2, 
fakefacts, 
waterfordwhispers news

### model

roberta-base

### results

stage_1: 全部数据, 1epoch

    dev：f1-macro = 0.5768
    test: f1-macro = 0.5709

stage_2: 全部数据, 10epoch

    dev：f1-macro = 0.6147
    test: f1-macro = 0.6190
