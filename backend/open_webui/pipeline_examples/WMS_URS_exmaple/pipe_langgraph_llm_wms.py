"""
title: LangGraph WMS需求分析助手
author: OpenWebUI Developer
version: 0.1.0
license: MIT
requirements: langgraph>=0.0.15, langchain-openai>=0.0.1, pydantic>=2.11.0
environment_variables: [OPENAI_API_KEY, AZURE_ENDPOINT, OPENAI_API_VERSION, MODEL_NAME]
Supports: [基础处理, 流式响应, LangGraph集成, WMS需求分析]
"""

import os
import logging
import asyncio
import time
import uuid
from typing import Dict, List, Tuple, Any, TypedDict, Annotated, Literal, Union, AsyncGenerator, Optional, Callable
from pydantic import BaseModel, Field

# LangGraph 和 LangChain 导入
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import AzureChatOpenAI

# 配置日志记录
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    stage: Literal["input", "analysis", "feedback", "complete"]
    # 当前反馈内容
    feedback: str

class Pipe:
    """
    基于 LangGraph 的WMS需求分析助手，使用 Azure OpenAI 服务
    
    该Pipe实现了一个WMS（仓库管理系统）需求分析助手，可以帮助用户完成WMS系统的需求收集和分析。
    系统会动态评估用户提供的需求信息，计算关键字段的覆盖度，并提供针对性的反馈和引导。
    
    主要功能：
    - 需求覆盖度评估：自动计算用户提供的需求中覆盖了多少WMS关键字段
    - 动态引导策略：根据缺失的字段提供针对性的反馈
    - 流式响应支持：支持实时流式输出分析结果
    - 状态追踪：跟踪整个需求分析的进度和状态
    """
    # 核心属性
    type = "pipe"
    id = "wms_requirements_analyzer"
    name = "WMS需求分析助手"

    # 配置类 (Valves)
    class Valves(BaseModel):
        OPENAI_API_KEY: str = Field(
            default=os.getenv("OPENAI_API_KEY", "7218515241f04d98b3b5d9869a25b91f"), 
            description="Azure OpenAI API Key"
        )
        AZURE_ENDPOINT: str = Field(
            default=os.getenv("AZURE_ENDPOINT", "https://nnitasia-openai-01-ins.openai.azure.com/"),
            description="Azure OpenAI Endpoint URL"
        )
        OPENAI_API_VERSION: str = Field(
            default=os.getenv("OPENAI_API_VERSION", "2023-09-01-preview"),
            description="Azure OpenAI API Version"
        )
        MODEL_NAME: str = Field(
            default=os.getenv("MODEL_NAME", "NNITAsia-GPT-4o"),
            description="Azure OpenAI Deployment Name"
        )
        emit_interval: float = Field(
            default=0.5, 
            ge=0, 
            description="状态发送间隔 (秒)"
        )
        enable_status_indicator: bool = Field(
            default=True, 
            description="启用状态指示器"
        )
        DEBUG: bool = Field(
            default=False, 
            description="启用调试日志"
        )
        FORCE_STREAMING: Optional[bool] = Field(
            default=None,
            description="强制启用/禁用流式响应（None 表示遵循请求设置）"
        )
        STREAM_CHUNK_SIZE: int = Field(
            default=20,
            ge=1,
            description="流式响应时每个块的最大字符数"
        )

    def __init__(self):
        """初始化 WMS需求分析助手 Pipe"""
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.graph = None
        self.llm = None
        self.wms_graph = None
        self.sys_msg = None
        
        log.info(f"Pipe '{self.name}' (ID: {self.id}) 初始化.")
        
        if self.valves.DEBUG:
            log.setLevel(logging.DEBUG)
            log.debug(f"Pipe '{self.id}' 在DEBUG模式下运行.")
        
        # 配置验证
        if not self.valves.OPENAI_API_KEY:
            log.warning(f"Pipe '{self.id}': OPENAI_API_KEY 环境变量未设置.")
        
        if not self.valves.AZURE_ENDPOINT:
            log.warning(f"Pipe '{self.id}': AZURE_ENDPOINT 环境变量未设置.")
        
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
        log.info(f"已加载WMS关键字段清单: {len(self.wms_key_fields)}项")
        
        # 系统消息
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
  1. 字段完整度评估: [x%] (针对17个关键字段)  
  2. 已覆盖字段: [字段1, 字段2, ...]  
  3. 待补充字段: [字段3, 字段4, ...]  
  4. 分析与建议: [根据当前完整度和业务逻辑提供针对性建议]  
  5. 下一步引导: [针对优先级最高的缺失字段进行提问]"""
        )
        log.info("系统消息已配置")

    # ======================
    # LANGGRAPH 初始化
    # ======================
    def _init_langgraph(self):
        """初始化 LangGraph 组件"""
        if self.wms_graph is not None:
            log.debug("LangGraph 已初始化，跳过")
            return
            
        log.info("初始化 LangGraph 组件...")
        
        # 初始化 LLM
        self.llm = AzureChatOpenAI(
            openai_api_key=self.valves.OPENAI_API_KEY,
            azure_endpoint=self.valves.AZURE_ENDPOINT,
            deployment_name=self.valves.MODEL_NAME,
            openai_api_version=self.valves.OPENAI_API_VERSION,
        )
        log.info("LLM 已初始化")
        
        # 构建图
        self.wms_graph = self._build_graph()
        log.info("WMS需求分析图已构建并编译")

    # ======================
    # GRAPH NODES
    # ======================
    def _user_input_node(self, state: WMSRequirementsState):
        """处理用户输入节点"""
        log.info("处理用户输入节点...")
        
        # 更新阶段
        state["stage"] = "analysis"
        
        log.debug(f"用户输入处理完成，当前状态: {state}")
        return state

    def _analysis_node(self, state: WMSRequirementsState):
        """需求分析节点"""
        log.info("执行需求分析节点...")
        
        # 获取最新的用户消息
        latest_message = state["messages"][-1].content
        log.debug(f"分析最新消息: {latest_message[:100]}...")
        
        # 如果状态中已经有了覆盖度信息，并且不是初始值，则保留之前的分析结果
        if state.get("_analyzed", False) and state.get("coverage", 0) > 0:
            log.info(f"使用已有分析结果: 覆盖度 {state['coverage']}%, 缺失字段 {len(state['missing_fields'])}")
            state["stage"] = "feedback"
            return state
            
        # 准备分析提示
        analysis_prompt = f"""
        请分析以下WMS需求描述，评估其覆盖了哪些WMS关键字段，并计算覆盖度：
        
        需求描述:
        {latest_message}
        
        WMS关键字段清单:
        {', '.join(self.wms_key_fields)}
        
        请提供:
        1. 已覆盖字段列表
        2. 覆盖度百分比
        3. 缺失字段列表
        
        以JSON格式返回，确保JSON格式正确无误:
        {{
            "covered_fields": ["字段1", "字段2", ...],
            "coverage_percentage": X,
            "missing_fields": ["字段3", "字段4", ...]
        }}
        
        只返回JSON对象，不要有其他文字说明。
        """
        
        # 调用LLM进行分析
        analysis_messages = [
            self.sys_msg,
            HumanMessage(content=analysis_prompt)
        ]
        
        analysis_response = self.llm.invoke(analysis_messages)
        log.debug(f"分析响应: {analysis_response.content}")
        
        # 解析响应中的JSON
        import json
        import re
        
        # 尝试直接解析整个响应
        try:
            # 清理响应文本，去除可能干扰JSON解析的字符
            cleaned_content = analysis_response.content.strip()
            # 尝试查找JSON对象的开始和结束
            start_idx = cleaned_content.find('{')
            end_idx = cleaned_content.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = cleaned_content[start_idx:end_idx+1]
                analysis_result = json.loads(json_str)
                
                # 更新状态
                state["coverage"] = analysis_result.get("coverage_percentage", 0)
                state["missing_fields"] = analysis_result.get("missing_fields", self.wms_key_fields.copy())
                state["_analyzed"] = True  # 标记已经分析过
                
                log.info(f"分析完成: 覆盖度 {state['coverage']}%, 缺失字段 {len(state['missing_fields'])}")
            else:
                log.warning("无法在响应中找到JSON对象")
                # 使用默认值
                state["coverage"] = 0
                state["missing_fields"] = self.wms_key_fields.copy()
        except json.JSONDecodeError as e:
            log.error(f"无法解析分析结果JSON: {e}")
            # 使用默认值
            state["coverage"] = 0
            state["missing_fields"] = self.wms_key_fields.copy()
        except Exception as e:
            log.error(f"分析过程中发生错误: {e}")
            # 使用默认值
            state["coverage"] = 0
            state["missing_fields"] = self.wms_key_fields.copy()
        
        # 更新阶段
        state["stage"] = "feedback"
        
        return state

    def _generate_feedback_node(self, state: WMSRequirementsState):
        """生成反馈节点"""
        log.info("生成反馈节点...")
        
        # 准备反馈提示
        feedback_prompt = f"""
        基于用户的WMS需求描述，我需要你生成反馈和下一步引导。
        
        当前状态:
        - 覆盖度: {state['coverage']}%
        - 已覆盖字段: {[field for field in self.wms_key_fields if field not in state['missing_fields']]}
        - 缺失字段: {state['missing_fields']}
        
        用户最新输入:
        {state['messages'][-1].content}
        
        请生成:
        1. 对当前需求的评价
        2. 已覆盖内容的摘要
        3. 针对缺失字段的引导性问题
        
        根据覆盖度调整响应:
        - 覆盖度<30%: 提供基础WMS概念和关键模块引导
        - 覆盖度30-70%: 针对缺失模块提供具体业务场景引导
        - 覆盖度>70%: 关注细节和业务流程闭环
        - 覆盖度>80%: 提供完整总结和建议
        """
        
        # 调用LLM生成反馈
        feedback_messages = [
            self.sys_msg,
            HumanMessage(content=feedback_prompt)
        ]
        
        feedback_response = self.llm.invoke(feedback_messages)
        log.debug(f"反馈响应: {feedback_response.content}")
        
        # 更新状态
        state["feedback"] = feedback_response.content
        state["messages"].append(AIMessage(content=feedback_response.content))
        
        return state

    def _route_by_coverage(self, state: WMSRequirementsState):
        """根据覆盖度决定下一步"""
        log.info(f"根据覆盖度路由，当前覆盖度: {state['coverage']}%")
        
        # 确保覆盖度是一个有效的数值
        try:
            coverage = float(state["coverage"])
        except (ValueError, TypeError):
            coverage = 0
            state["coverage"] = 0
        
        # 如果覆盖度达到100%，标记为完成
        if coverage >= 80:
            log.info("覆盖度达到100%，标记为完成")
            state["stage"] = "complete"
            return "complete"
        
        # 否则等待用户输入
        log.info("等待用户输入")
        state["stage"] = "input"
        return "wait_for_input"

    # ======================
    # GRAPH CONSTRUCTION
    # ======================
    def _build_graph(self):
        """构建并配置LangGraph状态机"""
        log.info("构建状态图...")
        
        # 创建状态图构建器
        builder = StateGraph(WMSRequirementsState)
        
        # 定义节点
        builder.add_node("user_input", self._user_input_node)
        builder.add_node("analysis", self._analysis_node)
        builder.add_node("generate_feedback", self._generate_feedback_node)
        log.info("节点已添加到图中")
        
        # 定义边
        builder.add_edge(START, "user_input")
        builder.add_edge("user_input", "analysis")
        builder.add_edge("analysis", "generate_feedback")
        
        # 添加条件边
        builder.add_conditional_edges(
            "generate_feedback",
            self._route_by_coverage,
            {
                "continue": "analysis",     # 继续处理当前输入
                "wait_for_input": END,     # 等待用户输入
                "complete": END            # 完成流程
            }
        )
        log.info("边已配置到图中")
        
        # 编译图
        compiled_graph = builder.compile()
        log.info("图编译完成")
        return compiled_graph

    # ======================
    # PIPE INTERFACE
    # ======================
    def pipes(self):
        """返回此 Pipe 提供的模型/功能列表"""
        return [
            {
                "id": f"{self.id}/default",
                "name": self.name,
                "context_length": 16000,
                "supports_vision": False,
                "supports_functions": False
            }
        ]

    async def pipe(self, body: Dict, **kwargs):
        """
        处理请求并返回响应
        
        Args:
            body: 请求体，包含消息和配置
            **kwargs: 额外参数，包括事件发射器
            
        Returns:
            流式或非流式响应
        """
        emitter = kwargs.get("__event_emitter__")
        user = kwargs.get("__user__")
        
        log.debug(f"Pipe '{self.id}' 收到请求. Body: {body}, User: {user}")
        await self.emit_status(emitter, "info", "开始处理...")
        
        try:
            # 1. 验证配置
            if not self.valves.OPENAI_API_KEY or not self.valves.AZURE_ENDPOINT:
                await self.emit_status(emitter, "error", "配置错误: API密钥或端点缺失", True)
                return {"error": "配置错误: API密钥或端点缺失"}
                
            # 2. 处理输入
            messages = self._process_messages(body.get("messages", []))
            model_id = body.get("model")
            
            # 确定是否使用流式响应
            is_stream = body.get("stream", False)
            if self.valves.FORCE_STREAMING is not None:
                is_stream = self.valves.FORCE_STREAMING
                
            log.debug(f"处理模型: {model_id}, 流式: {is_stream}")
            
            # 3. 延迟初始化LangGraph (只在第一次请求时创建)
            await self.emit_status(emitter, "info", "初始化LangGraph...")
            self._init_langgraph()
            
            # 4. 准备输入状态
            # 获取最新的用户消息
            latest_message = messages[-1].get("content", "")
            message_id = kwargs.get("__message_id__", str(uuid.uuid4()))
            
            # 创建初始消息
            llm_messages = [HumanMessage(content=latest_message)]
            
            # 准备输入状态
            previous_state = body.get("previous_state")
            
            # 生成一个稳定的消息ID，基于消息内容而非随机生成
            # 这样即使后台多次调用，也能保持相同的消息ID
            import hashlib
            content_hash = hashlib.md5(latest_message.encode()).hexdigest()
            message_id = f"msg_{content_hash}"
            
            # 检查是否已经处理过这条消息
            if previous_state and previous_state.get("last_processed_message_id") == message_id:
                log.info(f"消息ID {message_id} 已处理过，直接返回之前的结果")
                # 直接返回之前的结果
                final_message = previous_state.get("last_response", "已处理过的消息")
                # 如果是流式请求，需要返回流式响应
                if body.get("stream", False):
                    async def stream_cached_response():
                        # 分块返回缓存的响应
                        chunk_size = self.valves.STREAM_CHUNK_SIZE
                        for i in range(0, len(final_message), chunk_size):
                            chunk = final_message[i:i+chunk_size]
                            yield chunk
                            await asyncio.sleep(0.01)
                        # 发送状态信息
                        yield {"type": "state", "content": previous_state}
                    return stream_cached_response()
                else:
                    # 非流式请求直接返回
                    return {
                        "content": final_message,
                        "format": "text",
                        "state": previous_state
                    }
                
            # 如果有之前的状态，保留分析标记
            _analyzed = False
            if previous_state is not None:
                _analyzed = previous_state.get("_analyzed", False)
                
            input_state = {
                "messages": llm_messages,
                "coverage": 0 if previous_state is None else previous_state.get("coverage", 0),
                "missing_fields": self.wms_key_fields.copy() if previous_state is None else previous_state.get("missing_fields", self.wms_key_fields.copy()),
                "stage": "input",
                "feedback": "",
                "last_processed_message_id": message_id,
                "_analyzed": _analyzed  # 保留分析标记
            }
            
            # 5. 调用图
            await self.emit_status(emitter, "info", "处理需求分析...")
            
            if is_stream:
                # 返回流式响应
                log.debug("启动流式响应...")
                return self._stream_graph_updates(input_state, emitter)
            else:
                # 非流式响应
                log.debug("生成非流式响应...")
                result = self.wms_graph.invoke(input_state)
                
                # 获取最终消息
                final_message = result["messages"][-1].content
                log.debug(f"生成的非流式响应: {final_message[:100]}...")
                
                await self.emit_status(emitter, "info", "处理完成.", True)
                
                # 保存最后处理的消息ID和响应
                result["last_processed_message_id"] = message_id
                result["last_response"] = final_message
                
                # 返回结果和更新后的状态 - 需要先转换为可序列化的格式
                serializable_result = self._make_serializable(result)
                return {
                    "content": final_message, 
                    "format": "text",
                    "state": serializable_result  # 包含状态以便下次请求使用
                }
                
        except asyncio.TimeoutError:
            log.error(f"Pipe '{self.id}' 在 {self.valves.TIMEOUT}s 后超时", exc_info=True)
            await self.emit_status(emitter, "error", "请求超时", True)
            return {"error": "请求超时，请稍后再试"}
            
        except Exception as e:
            log.error(f"Pipe '{self.id}' 处理过程中发生错误: {e}", exc_info=True)
            await self.emit_status(emitter, "error", f"发生意外错误: {e}", True)
            return {"error": "发生意外错误，请查看日志"}

    async def _stream_graph_updates(self, graph_input: Dict, emitter: Optional[Callable]):
        """
        流式处理 LangGraph 更新
        
        Args:
            graph_input: LangGraph 输入
            emitter: 事件发射器
            
        Yields:
            流式响应块
        """
        try:
            log.info("开始流式处理 LangGraph 更新")
            await self.emit_status(emitter, "info", "开始流式处理...")
            
            # 创建队列用于线程间通信
            queue = asyncio.Queue()
            
            # 从输入状态中获取消息ID
            message_id = graph_input.get("last_processed_message_id", "unknown")
            
            # 在单独的线程中运行图
            async def stream_in_thread():
                try:
                    log.debug("在线程中启动图执行")
                    # 调用图
                    result = self.wms_graph.invoke(graph_input)
                    
                    # 获取最终消息
                    final_message = result["messages"][-1].content
                    log.debug(f"图执行完成，最终消息: {final_message[:100]}...")
                    
                    # 将结果分块发送到队列
                    chunk_size = self.valves.STREAM_CHUNK_SIZE
                    for i in range(0, len(final_message), chunk_size):
                        chunk = final_message[i:i+chunk_size]
                        await queue.put(chunk)
                        await asyncio.sleep(0.01)  # 小延迟模拟流式效果
                    
                    # 保存最后处理的消息ID和响应
                    result["last_processed_message_id"] = message_id
                    result["last_response"] = final_message
                    
                    # 将状态信息放入队列 - 需要先转换为可序列化的格式
                    serializable_result = self._make_serializable(result)
                    # 将状态信息放入队列，但使用特殊标记避免重复处理
                    await queue.put({"type": "state", "content": serializable_result})
                    
                    # 标记流结束
                    log.debug("流完成，标记结束")
                    await queue.put(None)
                except Exception as e:
                    log.error(f"流线程中出错: {e}", exc_info=True)
                    await queue.put({"type": "error", "content": f"流错误: {e}"})
                    await queue.put(None)
            
            # 启动流处理任务
            task = asyncio.create_task(stream_in_thread())
            
            # 从队列中获取结果并产生
            state_sent = False
            
            while True:
                chunk = await queue.get()
                if chunk is None:
                    # 流结束
                    log.debug("到达流结束")
                    break
                elif isinstance(chunk, dict):
                    if chunk.get("type") == "error":
                        # 错误情况
                        log.error(f"流中错误: {chunk['content']}")
                        yield chunk  # 错误保持字典格式
                        await self.emit_status(emitter, "error", chunk["content"], True)
                        break
                    elif chunk.get("type") == "state" and not state_sent:
                        # 保存状态信息，并在流结束时发送
                        log.debug("接收到状态信息")
                        state_sent = True
                        yield chunk  # 发送状态信息
                        continue
                else:
                    # 正常块 - 直接返回字符串，符合 OpenWebUI 流式格式
                    log.debug(f"产生块: {chunk[:50]}...")
                    yield chunk  # 直接返回字符串
            
            # 状态信息已经在循环中发送，这里不需要重复发送
            log.debug("状态信息已在流处理中发送")
            
            # 等待任务完成
            await task
            await self.emit_status(emitter, "info", "流完成.", True)
            log.info("流处理成功完成")
            
        except Exception as e:
            log.error(f"流式处理期间出错: {e}", exc_info=True)
            yield {"type": "error", "content": f"流错误: {e}"}
            await self.emit_status(emitter, "error", f"流错误: {e}", True)

    async def emit_status(self, emitter: Optional[Callable], level: str, message: str, done: bool = False):
        """
        发送状态更新
        
        Args:
            emitter: 事件发射器函数
            level: 状态级别 ('info', 'warning', 'error', 'critical')
            message: 状态消息
            done: 是否完成
        """
        if not emitter or not self.valves.enable_status_indicator:
            log.debug(f"状态发送跳过 (emitter: {bool(emitter)}, enabled: {self.valves.enable_status_indicator})")
            return
            
        current_time = time.time()
        # 除非是最终的 'done' 消息，否则限制发送频率
        if not done and (current_time - self.last_emit_time < self.valves.emit_interval):
            log.debug(f"状态发送节流: '{message}'")
            return
            
        try:
            status_payload = {
                "type": "status", 
                "level": level,
                "message": message,
                "done": done
            }
            log.debug(f"发送状态: {status_payload}")
            await emitter(status_payload)
            self.last_emit_time = current_time
        except Exception as e:
            # 记录错误但不中断管道
            log.error(f"发送状态失败 (level='{level}', msg='{message}'): {e}")

    def _process_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        处理消息格式
        
        Args:
            messages: OpenWebUI 格式的消息列表
            
        Returns:
            处理后的消息列表
        """
        processed = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            
            # 简单验证/转换
            if role and content:
                # 去除可能导致重复分析的空白字符
                content = content.strip()
                processed.append({"role": role, "content": content})
            else:
                log.warning(f"跳过无效的消息格式: {msg}")
                
        log.debug(f"处理的消息 ({len(processed)} 总计): {processed if self.valves.DEBUG else '...'}")
        return processed
        
    def _make_serializable(self, data):
        """
        将数据转换为可JSON序列化的格式
        
        Args:
            data: 可能包含非序列化对象的数据
            
        Returns:
            可序列化的数据结构
        """
        if isinstance(data, dict):
            return {k: self._make_serializable(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._make_serializable(item) for item in data]
        elif hasattr(data, "__dict__"):
            # 对于自定义对象，转换为字典
            return {"_type": data.__class__.__name__, "content": self._make_serializable(data.__dict__)}
        elif hasattr(data, "content") and hasattr(data, "type"):
            # 处理LangChain消息对象
            return {
                "_type": data.__class__.__name__,
                "role": data.type,
                "content": data.content
            }
        elif hasattr(data, "content"):
            # 处理其他带content属性的对象
            return {
                "_type": data.__class__.__name__,
                "content": data.content
            }
        else:
            # 基本类型直接返回
            return data

# 不需要创建实例，直接使用 Pipe 类
