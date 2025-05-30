# fake_news
随着互联网的迅猛发展，文本数据呈爆炸式增长，如何高效地组织并快速定位用户感兴趣的信息已成为自然语言处
理（NLP）领域的重要研究课题。文本分类作为信息检索与数据挖掘的核心技术之一，通过从原始文本中抽取语义特征并预
测其所属类别，可有效缓解信息过载问题。本文面向中文新闻场景，构建了一个基于深度学习的新闻文本自动分类系统。首
先，整理并清洗包含财经、房产、教育、科技、军事、汽车、体育、游戏、娱乐及其他共十个主题的大规模新闻语料；随后，采用
预训练语言模型BERT 对语料进行编码，并在下游分类任务上进行微调；最后，结合Softmax 分类器输出预测结果，并设计交
互式可视化界面以提升用户体验。实验结果表明，所提系统在测试集上取得94% 的总体准确率；其中体育、游戏及财经类新
闻的F1 值均超过96%，验证了模型在多类别不均衡场景下的优良泛化能力。该研究为中文新闻推荐与个性化信息服务提供
了可行方案，也为后续多模态新闻理解研究奠定了基础
