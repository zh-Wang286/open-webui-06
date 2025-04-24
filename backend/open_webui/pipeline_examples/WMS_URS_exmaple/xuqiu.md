# WMS-URS 需求文档生成系统改进需求

任务素材
@pipeline_examples/WMS-URS-exmaple.py/langgraph_llm_wms.py 
@pipeline_examples/WMS-URS-exmaple.py/CS-002-URS-01-WMS系统用户需求说明-20210723.txt 

## 背景
当前已有一个基于LangGraph的WMS(仓储管理系统)用户需求说明(URS)生成程序，能够通过对话交互引导用户补全需求信息。系统目前包含基础交互流程，但缺乏与标准URS文档的知识库整合以及自动文档生成能力。

## 目的
1. 集成标准URS文档知识库(RAG)提升需求收集质量
2. 增加自动文档生成功能，当需求覆盖度达标时自动输出标准格式URS文档
3. 优化系统决策流程，实现从交互收集到文档生成的完整闭环

## 功能变更

### 新增功能
1. **RAG知识库构建**
   - 将`CS-002-URS-01-WMS系统用户需求说明-20210723.txt`处理为可检索的知识库
   - 支持在文档生成阶段进行知识查询和格式规范校验

2. **RAG查询节点**
   - 输入：分析节点判定覆盖度≥80%的需求信息
   - 处理流程：
     1. 将需求拆解为3-5个核心需求点
     2. 对每个需求点检索RAG知识库`用户输入的需求：<实际用户输入> \n\n '='*20 \n\n 知识库召回片段：<片段内容1><片段内容2><片段内容3> `，然后格式化为URS的样子
     ```
    {
        "category": "功能性需求",
        "item_id": "URS-6.2-4",
        "item_name": "上架管理",
        "criticality": "关键",
        "content": "1、货架采用托盘管理和货位管理两种形式，使用扫描工具，托盘和货位要关联收货信息。上架完成，在系统可以查看每个物料放置的位置，能快速找到物料。\n2、所有物料完成上架后，回传给EAS生成库存。\n3、寄售仓的物料完成上架后，不回传给EAS生成库存，但是要传检验委托单给LIMS系统。"
      },
      ```
   - 输出：多条URS

3. **文档编写节点**
   - 输入：接收RAG查询节点的需求信息
   - 处理流程：
     1. 合并所有需求点
     2. 按照`"toc":["1.目的","2.范围","3.定义","4.参考资料","5.系统描述","6.详细要求","7.作者信息","8.附件","9.修订历史"],`重构为完整文档
   - 输出：符合企业规范的URS文档

### 流程变更
当前LangGraph结构
```
graph TD
    __start__([__start__])
    user_input(user_input)
    analysis(analysis)
    generate_feedback(generate_feedback)
    __end__([__end__])

    __start__ --> user_input
    user_input --> analysis
    analysis --> generate_feedback
    generate_feedback -- continue --> analysis
    generate_feedback -- wait_for_input --> __end__
    generate_feedback -- complete --> __end__
```


增加需求后的LangGraph结构
```
graph TD
    __start__([__start__])
    user_input(user_input)
    analysis(analysis)
    generate_feedback(generate_feedback)
    document_writer(document_writer)
    __end__([__end__])
    
    __start__ --> user_input
    user_input --> analysis
    analysis -->|coverage < 80%| generate_feedback
    generate_feedback -- continue --> analysis
    
    analysis -->|coverage ≥ 80%| rag_query
    rag_query --> document_writer_concat
    document_writer_concat --> __end__
```

## 技术栈
| 组件               | 版本       |
|--------------------|------------|
| langchain          | 0.3.19     |
| langchain-community| 0.3.18     |
| chromadb           | 0.6.2      |
| langgraph          | 0.2.50     |
| pydantic           | 2.11.3     |
```

任务：
1、读取`CS-002-URS-01-WMS系统用户需求说明-20210723.txt`文档格式，识别数据类型，合理切片，建立一个RAG系统。AzureOpenAIEmbedding
2、根据需求修改 `langgraph_llm_wms.py` 

