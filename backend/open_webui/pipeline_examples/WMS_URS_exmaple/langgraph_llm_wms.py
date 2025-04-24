# wms_requirements_analyzer.py
from typing import Dict, List, Tuple, Any, TypedDict, Annotated, Literal
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
import logging
import json
from pydantic import BaseModel, Field

# 导入 RAG 系统
from rag_system import WMSRAGSystem

# 定义状态类型
class WMSRequirementsState(TypedDict):
    """WMS需求分析状态"""
    # 消息历史
    messages: List[Any]
    # 当前WMS需求覆盖度 (0-100)
    coverage: int
    # 当前缺失的关键字段
    missing_fields: List[str]
    # 当前阶段
    stage: Literal["input", "analysis", "feedback", "rag_query", "document_writer", "complete"]
    # 当前反馈内容
    feedback: str
    # 核心需求点列表 (用于 RAG 查询)
    requirement_points: List[str]
    # RAG 查询结果
    rag_results: Dict[str, Any]
    # 生成的文档内容
    document_content: str

class WMSRequirementsAnalyzer:
    def __init__(self):
        # 配置日志
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("初始化WMS需求分析系统...")
        
        # ======================
        # API配置
        # ======================
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "7218515241f04d98b3b5d9869a25b91f")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT", "https://nnitasia-openai-01-ins.openai.azure.com/")
        self.openai_api_version = os.getenv("OPENAI_API_VERSION", "2023-09-01-preview")
        self.model_name_gpt_4o = os.getenv("MODEL_NAME", "NNITAsia-GPT-4o")
        
        self.logger.info("API配置已加载")
        
        # ======================
        # RAG系统配置
        # ======================
        self.wms_doc_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "CS-002-URS-01-WMS系统用户需求说明-20210723.txt"
        )
        self.rag_system = WMSRAGSystem(
            doc_path=self.wms_doc_path,
            azure_api_key=self.openai_api_key,
            azure_endpoint=self.azure_endpoint,
            azure_api_version=self.openai_api_version
        )
        # 构建向量数据库
        self.rag_system.build_vectorstore()
        self.logger.info("RAG系统已初始化")
        
        # ======================
        # WMS关键字段清单
        # ======================
        self.wms_key_fields = [
            "物料基础信息",
            "在库物料状态管理",
            "物料到库与入库管理",
            "上架管理",
            "抽样管理",
            "物料状态切换管理",
            "卡板管理",
            "在库管理",
            "出库管理",
            "盘点管理",
            "报表系统",
            "权限管理",
            "日志管理",
            "用户管理",
            "容错管理",
            "配置管理",
            "电子数据规范"
        ]
        self.logger.info(f"已加载WMS关键字段清单: {len(self.wms_key_fields)}项")
        
        # ======================
        # LLM设置
        # ======================
        self.llm = AzureChatOpenAI(
            openai_api_key=self.openai_api_key,
            azure_endpoint=self.azure_endpoint,
            deployment_name=self.model_name_gpt_4o,
            openai_api_version=self.openai_api_version,
        )
        self.logger.info("LLM已初始化")
        
        # ======================
        # 系统提示词
        # ======================
        self.sys_msg = SystemMessage(
            content="""- Role: WMS系统需求分析师与文档引导专家  
- Background: 仓库管理系统(WMS)需求文档需要覆盖全业务流程，需动态评估关键字段完整度并实施渐进式引导策略。  
- Profile: 具备WMS领域知识的需求引导系统，能同步处理字段完整性和业务逻辑闭环。  
- Skills:  
  1. WMS关键字段完整性评估（基础信息/业务流程/系统管理）  
  2. 动态引导策略（精准提示/业务场景联想/行业对标）  
  3. 构建WMS业务知识图谱（关联仓储规范、RFID标准、库存模型）  
  4. 实施三阶响应机制（字段提示→业务闭环→系统确认）  
- Goals:  
  1. 建立WMS需求收集的黄金标准（覆盖入库→在库→出库全流程）  
  2. 实现关键字段-业务逻辑的双向验证  
  3. 输出符合WMS行业规范的文档框架  
- Constrains:  
  1. 采用"业务流程树"模型管理对话分支（入库/存储/出库/支持系统）  
  2. 每个响应包含：字段完整度评估+引导策略说明（隐藏标签）  
  3. 关键节点进行业务逻辑确认（如状态切换与库存台账的关联性）  
- OutputFormat:  
  1. [字段状态] 当前WMS字段覆盖度(0-100%)及缺失业务环节  
  2. [引导策略] 当前采用的提示策略（核心字段追问/场景扩展/标准建议）  
  3. [响应内容] 直接输出需要补全的业务字段或确认请求  
  4. [内容风格] 专业、简洁、温和、不废话  

- **WMS关键字段清单**（仅供参考，疑似相近即可）：  
  ▶ 物料基础信息  
  ▶ 在库物料状态管理  
  ▶ 物料到库与入库管理  
  ▶ 上架管理  
  ▶ 抽样管理  
  ▶ 物料状态切换管理  
  ▶ 卡板管理  
  ▶ 在库管理  
  ▶ 出库管理  
  ▶ 盘点管理  
  ▶ 报表系统  
  ▶ 权限管理  
  ▶ 日志管理  
  ▶ 用户管理  
  ▶ 容错管理  
  ▶ 配置管理  
  ▶ 电子数据规范  

- Workflow:  
  1. 接收用户输入（WMS需求片段）  
  2. 执行字段完整性扫描（严格对照上述17项关键字段）  
  3. 评估覆盖度：  
     a) <50% → 提示缺失的核心业务字段（如"【上架管理】缺少策略定义（先进先出/动态货位）"）  
     b) 50-80% → 提示支持系统字段（如"【容错管理】需明确RFID读取失败的重试机制？"）  
     c) >80% → 输出"太棒啦(*^▽^*)"并启动最终验证  
  4. 缺失字段提示模板："当前缺少[字段名称]相关定义"""
        )
        self.logger.info("系统提示词已配置")
        
        # ======================
        # 图构建
        # ======================
        self.wms_graph = self._build_graph()
        self.logger.info("WMS需求分析图已构建并编译")
        
        self.logger.info("WMS需求分析系统初始化完成")

    # ======================
    # 节点函数
    # ======================
    def _user_input_node(self, state: WMSRequirementsState) -> WMSRequirementsState:
        """处理用户输入节点"""
        self.logger.info("处理用户输入节点...")
        
        # 这个节点主要是接收用户输入，不做特殊处理
        # 在实际应用中，这里可以添加用户输入的预处理逻辑
        
        # 重置阶段为分析阶段
        new_state = state.copy()
        new_state["stage"] = "analysis"
        
        return new_state

    def _analysis_node(self, state: WMSRequirementsState) -> WMSRequirementsState:
        """需求分析节点"""
        self.logger.info("处理需求分析节点...")
        
        # 准备消息列表
        messages = [self.sys_msg] + state["messages"]
        
        # 添加分析指令
        analysis_prompt = """
请分析用户提供的WMS需求信息，评估当前覆盖度，并按照以下格式输出：
1. 计算当前WMS字段覆盖度(0-100%)
2. 列出缺失的关键字段
3. 根据覆盖度决定下一步操作：
   - 如果覆盖度<50%，提示缺失的核心业务字段
   - 如果覆盖度在50-80%之间，提示支持系统字段
   - 如果覆盖度>80%，输出"太棒啦(*^▽^*)"

请在回复开头包含覆盖度百分比，格式为 [COVERAGE:XX] 其中XX为0-100的整数
请在回复中包含缺失字段列表，格式为 [MISSING:字段1,字段2,...]
"""
        messages.append(HumanMessage(content=analysis_prompt))
        
        # 调用LLM进行分析
        response = self.llm.invoke(messages)
        self.logger.debug(f"LLM分析响应: {response.content}")
        
        # 解析LLM响应中的覆盖度和缺失字段
        content = response.content
        coverage = 0
        missing_fields = []
        
        # 解析覆盖度
        import re
        coverage_match = re.search(r'\[COVERAGE:(\d+)\]', content)
        if coverage_match:
            coverage = int(coverage_match.group(1))
            self.logger.info(f"解析到覆盖度: {coverage}%")
        
        # 解析缺失字段
        missing_match = re.search(r'\[MISSING:(.*?)\]', content)
        if missing_match:
            missing_str = missing_match.group(1)
            if missing_str.strip():
                missing_fields = [field.strip() for field in missing_str.split(',')]
            self.logger.info(f"解析到缺失字段: {missing_fields}")
        
        # 更新状态
        new_state = state.copy()
        new_state["coverage"] = coverage
        new_state["missing_fields"] = missing_fields
        new_state["messages"] = state["messages"] + [response]
        
        return new_state

    def _generate_feedback_node(self, state: WMSRequirementsState) -> WMSRequirementsState:
        """生成反馈节点"""
        self.logger.info("处理生成反馈节点...")
        
        coverage = state["coverage"]
        missing_fields = state["missing_fields"]
        
        # 准备消息列表
        messages = [self.sys_msg] + state["messages"]
        
        # 添加反馈生成指令
        if coverage >= 80:
            feedback_prompt = """
太棒啦(*^▽^*)！WMS需求信息覆盖度已经很高了。
请生成一个最终的确认消息，表示需求收集已经完成，并准备进入下一阶段。
"""
            # 修改：将状态设置为rag_query而不是complete，以便路由到RAG查询节点
            new_stage = "rag_query"
            self.logger.info(f"覆盖度为{coverage}%，满足文档生成条件，设置阶段为rag_query")
        else:
            # 根据缺失字段生成提示
            fields_str = "、".join(missing_fields[:3]) if missing_fields else "关键字段"
            
            if coverage < 50:
                strategy = "核心字段追问"
                feedback_prompt = f"""
当前WMS需求信息覆盖度较低({coverage}%)，请针对缺失的核心业务字段({fields_str}等)生成详细的提示，
帮助用户补充这些关键信息。使用具体的业务示例和问题引导用户。
"""
            else:  # 50-80%
                strategy = "场景扩展/标准建议"
                feedback_prompt = f"""
当前WMS需求信息覆盖度中等({coverage}%)，请针对缺失的支持系统字段({fields_str}等)生成详细的提示，
帮助用户完善这些信息。可以使用行业标准或最佳实践作为参考。
"""
            new_stage = "feedback"
        
        messages.append(HumanMessage(content=feedback_prompt))
        
        # 调用LLM生成反馈
        response = self.llm.invoke(messages)
        self.logger.debug(f"LLM反馈响应: {response.content}")
        
        # 更新状态
        new_state = state.copy()
        new_state["messages"] = state["messages"] + [response]
        new_state["stage"] = new_stage
        new_state["feedback"] = response.content
        
        return new_state

    # ======================
    # 路由条件
    # ======================
    def _route_by_coverage(self, state: WMSRequirementsState) -> str:
        """根据覆盖度决定下一步"""
        coverage = state["coverage"]
        stage = state["stage"]
        
        self.logger.info(f"路由决策: 当前阶段={stage}, 覆盖度={coverage}%")
        
        if stage == "complete":
            self.logger.info(f"覆盖度为{coverage}%，流程完成")
            return "complete"
        elif stage == "rag_query":
            # 如果当前阶段已经是rag_query，直接进入RAG查询流程
            self.logger.info(f"当前阶段为rag_query，开始RAG查询流程")
            return "start_rag_query"
        elif stage == "feedback":
            self.logger.info(f"覆盖度为{coverage}%，需要用户继续输入")
            return "wait_for_input"
        else:
            self.logger.info(f"覆盖度为{coverage}%，继续处理")
            return "continue"

    # ======================
    # 图构建
    # ======================
    def _rag_query_node(self, state: WMSRequirementsState) -> WMSRequirementsState:
        """RAG查询节点，将需求拆解为核心需求点并检索知识库"""
        self.logger.info("处理RAG查询节点...")
        self.logger.info(f"当前状态: stage={state['stage']}, coverage={state['coverage']}%")
        
        # 准备消息列表
        messages = [self.sys_msg] + state["messages"]
        
        # 添加需求点拆解指令
        requirement_points_prompt = """
请将用户的WMS需求信息拆解为8-15个核心需求点，每个需求点应该简明扼要地描述一个关键功能或特性。

请以JSON格式输出需求点列表，格式为：
{"requirement_points": ["需求点1", "需求点2", "需求点3", ...]}

注意：每个需求点应该是一个独立的功能或特性，不要太宽泛也不要太具体。
"""
        messages.append(HumanMessage(content=requirement_points_prompt))
        
        # 调用LLM拆解需求点
        response = self.llm.invoke(messages)
        self.logger.debug(f"LLM需求点拆解响应: {response.content}")
        
        # 解析需求点
        try:
            import re
            import json
            # 尝试提取JSON部分
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                requirement_points_data = json.loads(json_str)
                requirement_points = requirement_points_data.get("requirement_points", [])
            else:
                # 如果没有找到JSON，尝试直接解析整个内容
                requirement_points_data = json.loads(response.content)
                requirement_points = requirement_points_data.get("requirement_points", [])
        except Exception as e:
            self.logger.error(f"解析需求点失败: {e}")
            # 如果解析失败，使用默认需求点
            requirement_points = ["WMS系统基础功能", "物料管理", "仓库操作流程"]
        
        self.logger.info(f"解析到需求点: {requirement_points}")
        
        # 对每个需求点进行RAG查询
        rag_results = {}
        for point in requirement_points:
            results = self.rag_system.query(point, top_k=3)
            rag_results[point] = results
            self.logger.info(f"需求点 '{point}' 查询到 {len(results)} 个相关结果")
        
        # 更新状态
        new_state = state.copy()
        new_state["requirement_points"] = requirement_points
        new_state["rag_results"] = rag_results
        new_state["stage"] = "document_writer"  # 直接设置为document_writer阶段，进入文档生成
        new_state["messages"] = state["messages"] + [response]
        
        self.logger.info(f"RAG查询完成，设置阶段为document_writer，找到{len(requirement_points)}个需求点和{sum(len(results) for results in rag_results.values())}个相关文档")
        
        return new_state
    
    def _document_writer_node(self, state: WMSRequirementsState) -> WMSRequirementsState:
        """文档编写节点，根据RAG查询结果生成完整的URS文档"""
        self.logger.info("处理文档编写节点...")
        
        # 准备消息列表
        messages = [self.sys_msg] + state["messages"]
        
        # 准备RAG查询结果
        rag_results_text = "\n\n你是URS文档专家，精通文档格式以及专业严谨的文风。以下是基于用户需求从WMS标准知识库中检索到的相关内容：\n\n"
        for point, results in state["rag_results"].items():
            rag_results_text += f"## 用户核心需求点: {point}\n"
            for i, result in enumerate(results, 1):
                rag_results_text += f"### 知识库相关文档 {i}:\n{result['content']}\n\n切记：当用户需求与知识库内容冲突时，优先以用户需求为准。"
        
        # 获取文档结构信息
        doc_structure = self.rag_system.get_document_structure()
        toc = doc_structure.get("toc", [])
        toc_text = "\n".join(toc)
        
        # 获取原始文档中的格式示例
        definition_format = """
{
  "abbr": "ER/ES",
  "explanation_zh": "电子记录电子签名",
  "explanation_en": null
}
"""
        
        requirement_format = """
{
  "category": "功能性需求",
  "item_id": "URS-6.2-4",
  "item_name": "上架管理",
  "criticality": "关键",
  "content": "1、货架采用托盘管理和货位管理两种形式，使用扫描工具，托盘和货位要关联收货信息。上架完成，在系统可以查看每个物料放置的位置，能快速找到物料。\n2、所有物料完成上架后，回传给EAS生成库存。\n3、寄售仓的物料完成上架后，不回传给EAS生成库存，但是要传检验委托单给LIMS系统。"
}
"""
        
        # 添加文档生成指令
        document_prompt = f"""
基于用户的需求和知识库检索结果，请生成一份完整的WMS系统用户需求说明(URS)文档。

文档应遵循以下目录结构：
{toc_text}

请确保文档内容：
1. 符合标准URS文档格式和规范
2. 整合用户提供的需求和知识库中的标准内容
3. 保持专业性和完整性
4. 使用JSON格式输出最终文档

知识库检索结果：
{rag_results_text}

请以JSON格式输出完整文档，格式应包含以下字段：
- doc_metadata: 文档元数据
- toc: 目录
- purpose: 目的
- scope: 范围
- description: 系统描述
- abbreviations: 缩写列表，每项必须严格按照以下格式：
{definition_format}
- requirements: 详细需求列表，每项必须严格按照以下格式：
{requirement_format}
- references: 参考资料
- authors: 作者信息

注意：请严格遵循上述格式要求，特别是缩写列表和需求列表的格式必须与示例完全一致。
"""
        messages.append(HumanMessage(content=document_prompt))
        
        # 调用LLM生成文档
        response = self.llm.invoke(messages)
        self.logger.debug(f"LLM文档生成响应: {response.content}")
        
        # 提取文档内容
        document_content = response.content
        
        # 更新状态
        new_state = state.copy()
        new_state["document_content"] = document_content
        new_state["stage"] = "complete"
        new_state["messages"] = state["messages"] + [response]
        
        self.logger.info("文档生成完成，设置阶段为complete")
        
        return new_state
    
    def _build_graph(self):
        """构建并配置LangGraph状态机"""
        self.logger.info("构建状态图...")
        
        # 创建状态图构建器
        builder = StateGraph(WMSRequirementsState)
        
        # 定义节点
        builder.add_node("user_input", self._user_input_node)
        builder.add_node("analysis", self._analysis_node)
        builder.add_node("generate_feedback", self._generate_feedback_node)
        builder.add_node("rag_query", self._rag_query_node)
        builder.add_node("document_writer", self._document_writer_node)
        self.logger.info("节点已添加到图中")
        
        # 定义边
        builder.add_edge(START, "user_input")
        builder.add_edge("user_input", "analysis")
        builder.add_edge("analysis", "generate_feedback")
        builder.add_edge("rag_query", "document_writer")
        builder.add_edge("document_writer", END)
        
        # 添加条件边
        builder.add_conditional_edges(
            "generate_feedback",
            self._route_by_coverage,
            {
                "continue": "analysis",     # 继续处理当前输入
                "wait_for_input": END,     # 等待用户输入
                "start_rag_query": "rag_query",  # 开始RAG查询
                "complete": END            # 完成流程
            }
        )
        
        # 添加详细日志
        self.logger.info("已配置generate_feedback节点的条件边：continue→analysis, wait_for_input→END, start_rag_query→rag_query, complete→END")
        self.logger.info("边已配置到图中")
        
        # 编译图
        compiled_graph = builder.compile()
        self.logger.info("图编译完成")
        return compiled_graph

    # ======================
    # 公共接口
    # ======================
    def run(self, query: str, previous_state=None):
        """执行WMS需求分析"""
        self.logger.info(f"处理查询: '{query}'")
        
        # 创建初始消息
        messages = [HumanMessage(content=query)]
        
        # 如果有之前的状态，合并消息
        if previous_state is not None and "messages" in previous_state:
            # 将之前的消息合并到当前消息中
            messages = previous_state["messages"] + messages
        
        # 准备输入状态
        input_state = {
            "messages": messages,
            "coverage": 0 if previous_state is None else previous_state.get("coverage", 0),
            "missing_fields": self.wms_key_fields.copy() if previous_state is None else previous_state.get("missing_fields", self.wms_key_fields.copy()),
            "stage": "input",
            "feedback": "",
            "requirement_points": [] if previous_state is None else previous_state.get("requirement_points", []),
            "rag_results": {} if previous_state is None else previous_state.get("rag_results", {}),
            "document_content": "" if previous_state is None else previous_state.get("document_content", "")
        }
        
        # 调用图
        result = self.wms_graph.invoke(input_state)
        
        # 获取最终消息和状态
        final_message = result["messages"][-1].content
        stage = result["stage"]
        
        # 如果是文档生成完成阶段，处理文档内容
        if stage == "complete" and "document_content" in result and result["document_content"]:
            try:
                # 尝试解析JSON文档
                import re
                import json
                # 先尝试查找JSON代码块
                json_match = re.search(r'```json\s*(.+?)\s*```', result["document_content"], re.DOTALL)
                if json_match:
                    document_json = json_match.group(1)
                else:
                    # 如果没有找到代码块，尝试直接解析整个内容
                    document_json = result["document_content"]
                
                # 尝试解析JSON并验证格式
                json_data = json.loads(document_json)
                
                # 验证并修正abbreviations格式
                if "abbreviations" in json_data and json_data["abbreviations"]:
                    for i, abbr in enumerate(json_data["abbreviations"]):
                        if not all(key in abbr for key in ["abbr", "explanation_zh", "explanation_en"]):
                            # 修正格式
                            corrected_abbr = {
                                "abbr": abbr.get("abbr", abbr.get("name", "")),
                                "explanation_zh": abbr.get("explanation_zh", abbr.get("description", "")),
                                "explanation_en": abbr.get("explanation_en", None)
                            }
                            json_data["abbreviations"][i] = corrected_abbr
                
                # 验证并修正requirements格式
                if "requirements" in json_data and json_data["requirements"]:
                    for i, req in enumerate(json_data["requirements"]):
                        if not all(key in req for key in ["category", "item_id", "item_name", "criticality", "content"]):
                            # 修正格式
                            corrected_req = {
                                "category": req.get("category", "功能性需求"),
                                "item_id": req.get("item_id", f"URS-{i+1}"),
                                "item_name": req.get("item_name", ""),
                                "criticality": req.get("criticality", "关键"),
                                "content": req.get("content", "")
                            }
                            json_data["requirements"][i] = corrected_req
                
                # 将修正后的JSON转回字符串
                document_json = json.dumps(json_data, ensure_ascii=False, indent=2)
                
                # 保存文档到文件
                output_file = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)),
                    "generated_wms_urs_document.json"
                )
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(document_json)
                self.logger.info(f"文档已保存到: {output_file}")
                final_message += f"\n\n文档已生成并保存到: {output_file}"
            except Exception as e:
                self.logger.error(f"保存文档失败: {e}")
                self.logger.error(f"文档内容: {result['document_content'][:200]}...")
        
        self.logger.info(f"最终结果: {final_message}")
        
        # 返回结果和更新后的状态
        return final_message, result

def interactive_session(analyzer):
    """运行交互式会话"""
    print("\nWMS需求分析系统 - 交互模式")
    print("输入'exit'或'quit'结束会话\n")
    print("已加载WMS关键字段知识库，当前待补全字段：17/17。请提供您的WMS需求描述：")
    
    # 保存会话状态
    current_state = None
    
    while True:
        try:
            query = input("您: ")
            
            if query.lower() in ['exit', 'quit', "q"]:
                print("再见!")
                break
                
            if not query.strip():
                print("请输入有效的查询")
                continue
                
            # 运行分析并获取结果和新状态
            result, current_state = analyzer.run(query, current_state)
            print(f"分析师: {result}")
            
            # 检查是否完成
            if current_state["stage"] == "complete":
                print("\n需求分析完成！")
                if "document_content" in current_state and current_state["document_content"]:
                    print("已生成WMS系统用户需求说明文档。")
                break
            
        except KeyboardInterrupt:
            print("\n再见!")
            break
        except Exception as e:
            logging.error(f"处理查询时出错: {e}")
            print(f"抱歉，发生错误: {e}")

if __name__ == "__main__":
    try:
        analyzer = WMSRequirementsAnalyzer()
        
        # 启动交互式会话
        interactive_session(analyzer)
        
    except Exception as e:
        logging.error(f"初始化分析器失败: {e}")
        print(f"初始化分析器失败: {e}")