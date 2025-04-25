#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
title: WMS需求分析系统Pipe
author: AI助手
version: 0.1.0
license: MIT
requirements: pydantic>=2.11.0, langchain_openai>=0.0.2, langchain_core>=0.1.15
environment_variables: [OPENAI_API_KEY, AZURE_ENDPOINT, OPENAI_API_VERSION]
supports: [需求分析, RAG检索, 文档生成]

本模块将LangGraph格式的WMS需求分析系统转换为OpenWebUI的Pipe格式。
主要功能包括：
1. 需求分析：分析用户输入，评估WMS需求覆盖度，提供反馈
2. RAG检索：根据需求点检索相关文档内容
3. 文档生成：根据分析结果生成URS文档
"""

import os
import json
import logging
import asyncio
import time
from datetime import datetime
from typing import (
    Dict,
    List,
    Union,
    AsyncGenerator,
    Optional,
    Callable,
    Any,
    TypedDict,
    Literal,
    Tuple,
)

from pydantic import BaseModel, Field
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 导入 RAG 系统
from open_webui.pipeline_examples.WMS_URS_exmaple.rag_system import WMSRAGSystem

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger(__name__)


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
    stage: Literal[
        "input", "analysis", "feedback", "rag_query", "document_writer", "complete"
    ]
    # 当前反馈内容
    feedback: str
    # 核心需求点列表 (用于 RAG 查询)
    requirement_points: List[str]
    # RAG 查询结果
    rag_results: Dict[str, Any]
    # 生成的文档内容
    document_content: str


class Pipe:
    """WMS需求分析系统管道，基于OpenWebUI的Pipe框架实现

    该类封装了WMS需求分析系统的所有功能，包括需求分析、RAG检索和文档生成。
    通过Pipe接口与OpenWebUI集成，支持异步处理和事件通知。

    主要组件：
    1. 需求输入与分析：分析用户输入，评估WMS需求覆盖度
    2. RAG检索：基于需求点检索相关WMS文档内容
    3. 文档生成：根据分析结果生成符合标准的URS文档

    属性：
        type: Pipe类型标识
        id: Pipe唯一标识符
        name: Pipe名称
        valves: 配置参数
        last_emit_time: 上次事件发送时间
        llm: LLM模型实例
        rag_system: RAG系统实例
        sys_msg: 系统提示词
        wms_key_fields: WMS关键字段清单
    """

    # 配置类 (Valves)
    class Valves(BaseModel):
        """WMS Pipe配置参数

        包含API配置、超时设置、事件发送设置等参数
        """

        OPENAI_API_KEY: Optional[str] = Field(
            default="7218515241f04d98b3b5d9869a25b91f",
            description="Azure OpenAI API 密钥",
        )
        AZURE_ENDPOINT: Optional[str] = Field(
            default="https://nnitasia-openai-01-ins.openai.azure.com/",
            description="Azure OpenAI 端点",
        )
        OPENAI_API_VERSION: Optional[str] = Field(
            default="2023-09-01-preview", description="OpenAI API 版本"
        )
        MODEL_NAME: Optional[str] = Field(
            default="NNITAsia-GPT-4o", description="GPT-4 模型名称"
        )
        TIMEOUT: int = Field(default=120, description="请求超时时间（秒）")
        emit_interval: float = Field(
            default=0.5, ge=0, description="状态发送间隔（秒）"
        )
        enable_status_indicator: bool = Field(
            default=True, description="启用状态指示器"
        )
        DEBUG: bool = Field(default=False, description="启用调试日志")
        WMS_DOC_PATH: Optional[str] = Field(default=None, description="WMS文档路径")

    # 核心属性
    type = "pipe"
    id = "wms_analyzer"
    name = "wms_analyzer"

    def __init__(self):
        """初始化WMS需求分析系统Pipe

        设置配置参数、初始化日志系统、加载WMS关键字段清单、
        初始化LLM模型和RAG系统等。
        """
        # 加载配置参数
        self.valves = self.Valves()
        self.last_emit_time = 0

        # 设置日志级别
        log.info(f"初始化 {self.name} (ID: {self.id})")
        if self.valves.DEBUG:
            log.setLevel(logging.DEBUG)
            log.debug(f"Pipe '{self.id}' 以DEBUG模式运行")

        # 验证关键配置
        if not self.valves.OPENAI_API_KEY:
            log.warning(f"Pipe '{self.id}': OPENAI_API_KEY环境变量未设置")
        if not self.valves.AZURE_ENDPOINT:
            log.warning(f"Pipe '{self.id}': AZURE_ENDPOINT环境变量未设置")

        # 设置WMS关键字段清单
        self._setup_wms_key_fields()

        # 初始化LLM模型
        self._init_llm()

        # 初始化RAG系统
        self._init_rag_system()

        # 设置系统提示词
        self._setup_system_prompt()

        log.info("WMS需求分析系统初始化完成")

    def _setup_wms_key_fields(self):
        """设置WMS关键字段清单"""
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
            "电子数据规范",
        ]
        log.info(f"已加载WMS关键字段清单: {len(self.wms_key_fields)}项")

    def _init_llm(self):
        """初始化LLM模型"""
        try:
            self.llm = AzureChatOpenAI(
                openai_api_key=self.valves.OPENAI_API_KEY,
                azure_endpoint=self.valves.AZURE_ENDPOINT,
                deployment_name=self.valves.MODEL_NAME,
                openai_api_version=self.valves.OPENAI_API_VERSION,
            )
            log.info("LLM模型初始化成功")
        except Exception as e:
            log.error(f"LLM模型初始化失败: {e}")
            raise

    def _init_rag_system(self):
        """初始化RAG系统"""
        try:
            # 设置文档路径
            if self.valves.WMS_DOC_PATH is None:
                self.valves.WMS_DOC_PATH = "/data01/open-webui-06/backend/open_webui/pipeline_examples/WMS_URS_exmaple/CS-002-URS-01-WMS系统用户需求说明-20210723.txt"

            # 初始化RAG系统
            self.rag_system = WMSRAGSystem(
                doc_path=self.valves.WMS_DOC_PATH,
                azure_api_key=self.valves.OPENAI_API_KEY,
                azure_endpoint=self.valves.AZURE_ENDPOINT,
                azure_api_version=self.valves.OPENAI_API_VERSION,
            )

            # 构建向量数据库
            self.rag_system.build_vectorstore()
            log.info("RAG系统初始化成功")
        except Exception as e:
            log.error(f"RAG系统初始化失败: {e}")
            raise

    def _setup_system_prompt(self):
        """设置系统提示词"""
        self.sys_msg = SystemMessage(
            content="""- Role: WMS系统需求分析师与文档引导专家  
- Background: 仓库管理系统(WMS)需求文档需要覆盖全业务流程，需动态评估关键字段完整度并实施渐进式引导策略。  
- Mission:  
  1. 收集WMS需求信息  
  2. 确保需求完整性（必须覆盖全部核心需求）  
  3. 生成符合质量要求的URS文档  
  
- WMS关键需求字段清单(17项):  
  物料基础信息, 在库物料状态管理, 物料到库与入库管理, 上架管理, 抽样管理, 物料状态切换管理, 卡板管理, 在库管理, 出库管理, 盘点管理, 报表系统, 权限管理, 日志管理, 用户管理, 容错管理, 配置管理, 电子数据规范
  
- 需求分析与引导策略:  
  1. 解析用户提供的需求描述  
  2. 执行字段完整性扫描（严格对照上述17项关键字段）  
  3. 评估覆盖度：  
   a) <50% → 提示缺失的核心业务字段（如“【上架管理】缺少策略定义（先进先出/动态货位）”）  
   b) 50-80% → 询问特定缺失详情（如“您对【报表系统】有什么特殊需求？”）  
   c) >80% → 转到需求文档生成步骤  

- 最终输出:  
  生成完整URS文档，包含：  
  1. 文档元数据（标题、版本等）  
  2. 目的与范围  
  3. 缩写说明  
  4. 需求列表（分类、编号、内容、重要性）
"""
        )
        log.debug("LLM系统提示词设置完成")

    def pipes(self) -> List[dict]:
        """返回此Pipe提供的模型/功能列表

        实现Pipe框架所需的pipes方法，定义并返回此Pipe提供的模型信息。

        返回：
            包含模型信息的字典列表
        """
        return [
            {
                "id": f"{self.id}/default",  # 模型唯一ID
                "name": self.name,  # 显示名称
                "context_length": 8192,  # 上下文窗口大小
                "supports_vision": False,  # 不支持视觉输入
                "supports_functions": False,  # 不支持函数调用
                "description": "WMS需求分析系统，分析需求完整性并生成URS文档",
            },
        ]

    async def pipe(
        self, body: Dict, **kwargs
    ) -> Union[str, Dict, AsyncGenerator[Union[str, Dict], None]]:
        """执行WMS需求分析

        实现Pipe框架所需的pipe方法，处理并响应客户端请求。
        支持流式和非流式响应。

        参数：
            body: 请求体，包含消息、模型、流式设置等
            **kwargs: 其他参数，包括__event_emitter__等

        返回：
            字符串、字典或异步生成器，取决于处理模式
        """
        # 提取事件发射器和其他上下文
        emitter = kwargs.get("__event_emitter__")
        metadata = kwargs.get("__metadata__")
        user = kwargs.get("__user__")

        # 记录请求信息
        log.info(
            f"Pipe '{self.id}' 收到请求。用户ID: {user['id'] if user else 'unknown'}"
        )
        log.debug(f"Pipe '{self.id}' 收到请求。Body: {body}")

        # 发送开始状态
        await self.emit_status(emitter, "info", "开始处理WMS需求分析...")

        try:
            # 1. 验证配置
            if not self.valves.OPENAI_API_KEY or not self.valves.AZURE_ENDPOINT:
                await self.emit_status(
                    emitter, "error", "配置错误: 缺少API密钥或端点配置", True
                )
                return {"error": "配置错误: 缺少API密钥或端点配置"}

            # 2. 处理输入
            messages = self._process_messages(body.get("messages", []))
            is_stream = body.get("stream", False)

            # 3. 获取当前状态 (不存在时创建初始状态)
            state = self._get_current_state(messages)

            # 4. 执行需求分析流程
            await self.emit_status(emitter, "info", "执行需求分析...")

            # 当前只有一个用户消息时，初始化分析
            if len(messages) == 1 and messages[0]["role"] == "user":
                # 初始化分析
                result, updated_state = await self._analyze_requirements(
                    state, messages[-1]["content"], emitter
                )
            else:
                # 继续分析
                result, updated_state = await self._continue_analysis(
                    state, messages[-1]["content"], emitter
                )

            # 5. 返回结果
            if is_stream:
                # 返回流式响应
                return self._stream_response(result, emitter)
            else:
                # 返回非流式响应
                await self.emit_status(emitter, "info", "分析完成", True)
                return {"content": result, "format": "text"}

        except asyncio.TimeoutError:
            log.error(f"Pipe '{self.id}' 超时，超过 {self.valves.TIMEOUT}s")
            await self.emit_status(emitter, "error", "请求超时", True)
            return {"error": "请求超时，请稍后重试"}
        except Exception as e:
            log.error(f"Pipe '{self.id}' 处理错误: {e}", exc_info=True)
            await self.emit_status(emitter, "error", f"发生错误: {e}", True)
            return {"error": "处理请求时发生错误，请查看日志"}

    def _get_current_state(self, messages: List[Dict]) -> WMSRequirementsState:
        """获取当前状态或创建初始状态

        根据消息历史获取当前状态，如果不存在则创建初始状态。

        参数：
            messages: 消息列表

        返回：
            WMSRequirementsState: 当前状态
        """
        # 初始化状态
        initial_state: WMSRequirementsState = {
            "messages": messages,
            "coverage": 0,
            "missing_fields": self.wms_key_fields.copy(),
            "stage": "input",
            "feedback": "",
            "requirement_points": [],
            "rag_results": {},
            "document_content": "",
        }

        # 如果消息列表为空，返回初始状态
        if not messages:
            log.debug("创建新的WMS需求分析状态")
            return initial_state

        # 检查最后一条AI消息中是否包含状态信息
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                if "__state__" in content:
                    try:
                        # 尝试提取并解析状态JSON
                        state_json = content.split("__state__")[1].strip()
                        state_data = json.loads(state_json)
                        log.debug("成功从消息历史中恢复状态")

                        # 确保所有必要字段存在
                        for key in initial_state.keys():
                            if key not in state_data:
                                state_data[key] = initial_state[key]

                        # 更新消息历史
                        state_data["messages"] = messages

                        return state_data
                    except Exception as e:
                        log.error(f"解析状态数据失败: {e}")
                        break

        # 如果未找到状态信息，返回初始状态
        log.debug("在消息历史中未找到状态信息，创建新状态")
        return initial_state

    def _process_messages(self, messages: List[Dict]) -> List[Dict]:
        """处理输入消息列表

        将OpenWebUI的消息列表转换为标准格式。

        参数：
            messages: OpenWebUI格式的消息列表

        返回：
            处理后的消息列表
        """
        processed = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            # 简单验证/转换
            if role and content:
                # 处理可能的“__state__”标记
                if role == "assistant" and "__state__" in content:
                    # 将状态信息从显示给用户的内容中移除
                    visible_content = content.split("__state__")[0].strip()
                    processed.append({"role": role, "content": visible_content})
                else:
                    processed.append({"role": role, "content": content})
            else:
                log.warning(f"跳过无效的消息格式: {msg}")

        log.debug(f"处理消息 ({len(processed)} 条)")
        return processed

    async def emit_status(
        self, emitter: Optional[Callable], level: str, message: str, done: bool = False
    ):
        """发送状态更新

        使用事件发射器向客户端发送状态更新。

        参数：
            emitter: 事件发射器函数
            level: 状态级别 (info, warning, error)
            message: 状态消息
            done: 是否完成
        """
        if not emitter or not self.valves.enable_status_indicator:
            return

        try:
            # 检查是否应该发送状态更新
            current_time = time.time()
            if (
                not done
                and current_time - self.last_emit_time < self.valves.emit_interval
            ):
                return

            # 发送状态更新
            await emitter(
                {
                    "type": "update:status",
                    "data": {"level": level, "content": message, "done": done},
                }
            )
            log.debug(f"状态更新已发送: [{level}] {message} (done={done})")
            self.last_emit_time = current_time
        except Exception as e:
            # 记录错误但不使管道崩溃
            log.error(f"发送状态更新失败 (level='{level}', msg='{message}'): {e}")

    async def _stream_response(
        self, content: str, emitter: Optional[Callable]
    ) -> AsyncGenerator[Dict, None]:
        """将响应内容以流式方式返回

        将完整响应内容分块返回，模拟流式响应。

        参数：
            content: 要流式返回的完整内容
            emitter: 事件发射器函数

        返回：
            异步生成器，产生响应块
        """
        log.debug("开始流式响应生成...")
        try:
            # 将内容按文本块分割
            chunks = [
                content[i : i + 10] for i in range(0, len(content), 10)
            ]  # 每10个字符一块

            # 逐块生成响应
            for i, chunk in enumerate(chunks):
                # 发送进度更新
                if i % 20 == 0 and emitter:  # 每20块发送一次进度更新
                    progress = int((i / len(chunks)) * 100)
                    await self.emit_status(emitter, "info", f"生成中: {progress}%")

                # 等待一小段时间模拟流式生成
                await asyncio.sleep(0.01)

                # 返回当前块
                # yield {"type": "chunk", "content": chunk}
                yield chunk

            # 发送完成状态
            if emitter:
                await self.emit_status(emitter, "info", "生成完成", True)

            log.debug("流式响应生成完成")

        except Exception as stream_err:
            log.error(f"流式响应生成错误: {stream_err}", exc_info=True)
            if emitter:
                await self.emit_status(
                    emitter, "error", f"流式响应错误: {stream_err}", True
                )
            yield {"type": "error", "content": f"流式响应错误: {stream_err}"}

    async def _analyze_requirements(
        self, state: WMSRequirementsState, user_input: str, emitter: Optional[Callable]
    ) -> Tuple[str, WMSRequirementsState]:
        """执行需求分析

        分析用户输入的需求描述，评估需求覆盖度，生成反馈。

        参数：
            state: 当前状态
            user_input: 用户输入的需求描述
            emitter: 事件发射器函数

        返回：
            Tuple[str, WMSRequirementsState]: 响应内容和更新后的状态
        """
        log.info("执行初始需求分析")
        await self.emit_status(emitter, "info", "分析需求覆盖度...")

        try:
            # 准备LLM输入
            messages = [
                self.sys_msg,
                HumanMessage(
                    content=f"""请分析以下用户提供的WMS需求描述，评估其需求覆盖度

用户需求描述: 
{user_input}

分析要求:
1. 根据17项WMS关键字段对需求进行分析。
2. 估算需求覆盖度百分比（0-100%）。
3. 列出已覆盖的字段和缺失的字段。
4. 提供下一步行动建议。

输出格式: JSON对象，包含以下字段:
{{
  "covered_fields": [已覆盖字段列表],
  "missing_fields": [缺失字段列表],
  "coverage_percentage": 数字(0-100),
  "analysis": "需求分析总结",
  "next_steps": "下一步建议"
}}"""
                ),
            ]

            # 调用LLM为用户分析需求
            log.debug("调用LLM分析需求覆盖度")
            ai_response = await self.llm.ainvoke(messages)
            log.debug(f"LLM响应: {ai_response.content}")

            # 解析LLM响应
            try:
                content = ai_response.content
                # 提取JSON字符串
                if "```json" in content and "```" in content:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                elif "{" in content and "}" in content:
                    # 尝试提取第一个完整的JSON对象
                    json_str = content[content.find("{") : content.rfind("}") + 1]
                else:
                    json_str = content

                # 解析JSON
                analysis_result = json.loads(json_str)

                # 更新状态
                state["coverage"] = analysis_result.get("coverage_percentage", 0)
                state["missing_fields"] = analysis_result.get("missing_fields", [])

                # 将JSON结果转换为友好的文本格式作为反馈
                feedback = (
                    f"## WMS需求分析结果\n\n"
                    f"### 需求覆盖度: {analysis_result.get('coverage_percentage')}%\n\n"
                    f"### 已覆盖字段 ({len(analysis_result.get('covered_fields', []))}):\n"
                    f"{', '.join(analysis_result.get('covered_fields', []))}\n\n"
                    f"### 缺失字段 ({len(analysis_result.get('missing_fields', []))}):\n"
                    f"{', '.join(analysis_result.get('missing_fields', []))}\n\n"
                    f"### 分析:\n{analysis_result.get('analysis', '')}\n\n"
                    f"### 下一步建议:\n{analysis_result.get('next_steps', '')}"
                )

                # 根据覆盖度决定下一阶段
                if state["coverage"] < 50:
                    state["stage"] = "feedback"  # 需要更多需求输入
                    next_action = "请提供更多需求信息，尤其是上面列出的缺失字段。"
                elif state["coverage"] >= 80:
                    state["stage"] = "rag_query"  # 可以转到RAG查询阶段
                    next_action = "您的需求已经覆盖了大部分关键字段，我们将基于此生成详细的URS文档。"
                else:
                    state["stage"] = "feedback"  # 需要针对性提问
                    next_action = "感谢您的需求描述，但还缺少一些关键详情。请充实上面列出的缺失字段。"

                # 生成最终响应
                response = f"{feedback}\n\n{next_action}"

                # 更新状态
                state["feedback"] = feedback

                # 如果进入RAG阶段，生成需求点
                if state["stage"] == "rag_query":
                    try:
                        await self.emit_status(
                            emitter, "info", "生成需求点以进行RAG检索..."
                        )
                        # 调用LLM生成需求点
                        state = await self._generate_requirement_points(
                            state, user_input, emitter
                        )
                        # 继续进行RAG查询
                        state, rag_response = await self._perform_rag_query(
                            state, emitter
                        )
                        response += f"\n\n{rag_response}"
                    except Exception as e:
                        log.error(f"RAG处理错误: {e}", exc_info=True)
                        response += f"\n\n注意: 生成详细文档时发生错误，但已保存您的需求分析结果。"

            except json.JSONDecodeError as e:
                log.error(f"JSON解析错误: {e}", exc_info=True)
                response = f"\n\n对不起，在处理需求分析结果时发生错误。请重试或提供更详细的需求描述。"
                state["stage"] = "input"

            # 返回响应和更新后的状态
            return response, state

        except Exception as e:
            log.error(f"需求分析错误: {e}", exc_info=True)
            error_response = f"对不起，在分析需求时发生错误: {e}"
            return error_response, state

    async def _generate_requirement_points(
        self, state: WMSRequirementsState, user_input: str, emitter: Optional[Callable]
    ) -> WMSRequirementsState:
        """生成需求点列表用于RAG查询

        使用LLM将用户的需求描述转换为积极性需求点列表，供后续RAG查询使用。

        参数：
            state: 当前状态
            user_input: 用户输入的需求描述
            emitter: 事件发射器函数

        返回：
            WMSRequirementsState: 更新后的状态
        """
        log.info("生成需求点列表")
        await self.emit_status(emitter, "info", "将需求分解为核心需求点...")

        try:
            # 准备LLM输入
            messages = [
                self.sys_msg,
                HumanMessage(
                    content=f"""请将以下用户提供的WMS需求描述分解为10-15个特定的需求点，以便进行知识检索。

用户需求描述: 
{user_input}

要求：
1. 将需求分解为清晰有焦点的需求点
2. 每个需求点应为一句简洁清晰的描述
3. 覆盖基本核心需求和特殊要求
4. 需求点应充分有明确的关键字，方便检索

输出格式: JSON列表，仅包含需求点字符串，格式如下:
["需求点1", "需求点2", ...]
"""
                ),
            ]

            # 调用LLM分解需求
            log.debug("调用LLM生成需求点")
            ai_response = await self.llm.ainvoke(messages)
            log.debug(f"LLM响应: {ai_response.content}")

            # 解析LLM响应
            content = ai_response.content
            # 提取JSON字符串
            if "```json" in content and "```" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content and "```" in content:
                json_str = content.split("```")[1].strip()
            elif "[" in content and "]" in content:
                # 尝试提取第一个完整的JSON数组
                json_str = content[content.find("[") : content.rfind("]") + 1]
            else:
                json_str = content

            # 解析JSON
            requirement_points = json.loads(json_str)

            # 验证结果
            if not isinstance(requirement_points, list):
                log.warning(f"需求点格式错误，不是列表: {requirement_points}")
                requirement_points = [
                    "WMS物料管理需求",
                    "仓库入库流程",
                    "仓库出库流程",
                    "库存管理",
                    "盘点管理",
                    "用户权限管理",
                    "报表系统需求",
                ]

            # 更新状态
            state["requirement_points"] = requirement_points
            log.info(f"生成了 {len(requirement_points)} 个需求点")

            return state

        except Exception as e:
            log.error(f"生成需求点错误: {e}", exc_info=True)
            # 如果出错，设置一些默认需求点
            default_points = [
                "WMS物料管理需求",
                "仓库入库流程",
                "仓库出库流程",
                "库存管理",
                "盘点管理",
                "用户权限管理",
            ]
            state["requirement_points"] = default_points
            await self.emit_status(
                emitter, "warning", f"生成需求点时出错，使用默认需求点"
            )
            return state

    async def _perform_rag_query(
        self, state: WMSRequirementsState, emitter: Optional[Callable]
    ) -> Tuple[WMSRequirementsState, str]:
        """执行RAG查询

        根据需求点列表查询相关的WMS文档内容，为文档生成提供参考。

        参数：
            state: 当前状态
            emitter: 事件发射器函数

        返回：
            Tuple[WMSRequirementsState, str]: 更新后的状态和响应内容
        """
        log.info("执行RAG查询")
        await self.emit_status(emitter, "info", "从知识库检索相关内容...")

        try:
            # 确保存在需求点
            if not state["requirement_points"]:
                log.warning("需求点列表为空，无法执行RAG查询")
                return state, "无法进行知识检索，需求点列表为空。"

            # 执行RAG查询
            log.debug(f"开始查询 {len(state['requirement_points'])} 个需求点")
            rag_results = self.rag_system.query_by_requirement_points(
                state["requirement_points"], top_k=3  # 每个需求点返回的结果数量
            )

            # 更新状态
            state["rag_results"] = rag_results
            state["stage"] = "document_writer"  # 转到文档生成阶段

            # 生成响应
            response = "相关知识库检索完成。正基于您的需求和知识库信息生成完整的WMS系统用户需求说明文档..."

            # 生成文档
            await self.emit_status(emitter, "info", "正在生成URS文档...")
            state, document_response = await self._generate_document(state, emitter)

            return state, response + "\n\n" + document_response

        except Exception as e:
            log.error(f"RAG查询错误: {e}", exc_info=True)
            state["stage"] = "feedback"  # 返回到反馈阶段
            await self.emit_status(emitter, "error", f"RAG查询错误: {e}")
            return (
                state,
                f"对不起，在检索相关知识时出错: {e}\n\n请提供更多的需求描述，或者重新尝试。",
            )

    async def _generate_document(
        self, state: WMSRequirementsState, emitter: Optional[Callable]
    ) -> Tuple[WMSRequirementsState, str]:
        """生成WMS系统用户需求说明文档

        根据RAG查询结果和用户需求生成完整的URS文档。

        参数：
            state: 当前状态
            emitter: 事件发射器函数

        返回：
            Tuple[WMSRequirementsState, str]: 更新后的状态和响应内容
        """
        log.info("生成WMS需求文档")
        await self.emit_status(emitter, "info", "生成WMS系统用户需求说明文档...")

        try:
            # 准备LLM输入
            # 格式化RAG结果作为上下文
            rag_content = self.rag_system.format_rag_results_for_llm(
                "".join(
                    [
                        msg["content"]
                        for msg in state["messages"]
                        if msg["role"] == "user"
                    ]
                ),
                [
                    result
                    for point_results in state["rag_results"].values()
                    for result in point_results
                ],
            )

            messages = [
                self.sys_msg,
                HumanMessage(
                    content=f"""请根据用户需求和知识库参考资料，生成标准的WMS系统用户需求说明(URS)文档。

知识库参考资料:
{rag_content}

文档要求:
1. 生成符合医药行业标准的WMS系统用户需求说明文档
2. 包含标准的任何URS需要有的结构和内容
3. 按照重要性分类用户需求(关键性、主要、次要)
4. 使用JSON格式返回文档

输出格式:
```json
{{
  "doc_metadata": {{
    "title": "WMS系统用户需求说明",
    "document_id": "URS-WMS-001",
    "version": "1.0",
    "date": "{datetime.now().strftime('%Y-%m-%d')}"
  }},
  "purpose": "文档目的说明",
  "scope": "适用范围说明",
  "toc": [目录结构],
  "abbreviations": [
    {{
      "abbr": "URS",
      "explanation_zh": "用户需求规格",
      "explanation_en": "User Requirement Specification"
    }},
    // 其他缩写说明...
  ],
  "requirements": [
    {{
      "category": "功能性需求",  // 功能性需求、非功能性需求、控制需求等
      "item_id": "URS-1",
      "item_name": "需求名称",
      "criticality": "关键",  // 关键、主要、次要
      "content": "需求具体描述"
    }},
    // 更多需求项...
  ]
}}
```
"""
                ),
            ]

            # 调用LLM生成文档
            log.debug("调用LLM生成需求文档")
            ai_response = await self.llm.ainvoke(messages)
            log.debug(f"LLM响应: {ai_response.content[:500]}...")

            # 提取JSON字符串
            content = ai_response.content
            if "```json" in content and "```" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content and "```" in content:
                json_str = content.split("```")[1].strip()
            elif "{" in content and "}" in content:
                # 尝试提取第一个完整的JSON对象
                json_str = content[content.find("{") : content.rfind("}") + 1]
            else:
                json_str = content

            # 解析JSON
            document_data = json.loads(json_str)

            # 更新状态
            state["document_content"] = json_str
            state["stage"] = "complete"  # 完成分析

            # 尝试保存文档到静态资源目录，使其可通过URL访问
            try:
                # 创建文档文件名（使用时间戳确保唯一性）
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                filename = f"wms_urs_document_{timestamp}.json"
                print(f"====================filename: {filename}====================")
                # 静态资源目录路径 - 使用相对于应用程序的路径
                # 获取backend/open_webui目录的路径
                backend_dir = "/data01/open-webui-06/backend/open_webui"

                # 记录当前目录信息便于调试
                log.info(f"当前文件路径: {__file__}")
                log.info(f"后端目录路径: {backend_dir}")

                # 尝试在当前目录创建 wms_docs 子目录
                current_dir = os.path.dirname(os.path.abspath(__file__))
                static_dir = os.path.join(current_dir, "wms_docs")

                # 确保目录存在
                try:
                    os.makedirs(static_dir, exist_ok=True)
                    log.info(f"创建或确认目录存在: {static_dir}")
                except Exception as e:
                    # 如果出现任何错误，则使用当前目录
                    static_dir = current_dir
                    log.warning(f"创建 wms_docs 子目录失败: {e}")
                    log.warning(f"使用当前目录: {static_dir}")

                # 完整的文件路径
                output_file = os.path.join(static_dir, filename)

                # 保存文件
                with open(output_file, "w", encoding="utf-8") as f:
                    f.write(json.dumps(document_data, ensure_ascii=False, indent=2))

                # 构建可访问的URL
                # 使用相对路径而不是绝对路径
                # 将文件复制到静态目录中，以确保可以通过URL访问
                try:
                    # 静态资源目录 - 确保使用正确的路径
                    # 创建临时目录用于存储文档
                    tmp_dir = "/tmp/wms_docs"
                    os.makedirs(tmp_dir, exist_ok=True)
                    log.info(f"创建或确认目录存在: {tmp_dir}")

                    # 复制文件到临时目录
                    tmp_file_path = os.path.join(tmp_dir, filename)
                    with open(tmp_file_path, "w", encoding="utf-8") as f:
                        f.write(json.dumps(document_data, ensure_ascii=False, indent=2))

                    # 尝试复制到静态目录 - 如果有权限的话
                    try:
                        # 正确的静态资源目录路径
                        static_resource_dir = os.path.join(
                            backend_dir, "static", "wms_docs"
                        )
                        os.makedirs(static_resource_dir, exist_ok=True)

                        # 复制文件到静态目录
                        static_file_path = os.path.join(
                            static_resource_dir, f"wms_urs_document_{timestamp}.json"
                        )
                        print(f"====================static_file_path: {static_file_path}====================")

                        with open(static_file_path, "w", encoding="utf-8") as f:
                            f.write(
                                json.dumps(document_data, ensure_ascii=False, indent=2)
                            )

                        log.info(f"文件已复制到静态目录: {static_file_path}")
                    except Exception as e:
                        log.warning(f"复制到静态目录失败，将使用临时文件: {e}")

                    # 设置正确的URL路径 - 始终使用固定文件名以便外部访问
                    url_path = f"/static/wms_docs/wms_urs_document_{timestamp}.json"
                    print(f"====================url_path: {url_path}====================")

                    # 考虑前后端分离的情况
                    full_url = f"http://chatgpt.nnit.cn:8180{url_path}"
                    log.info(f"文档可通过URL访问: {full_url}")
                except Exception as e:
                    log.error(f"复制文件到静态目录失败: {e}")
                    # 如果复制失败，使用原始文件路径
                    full_url = f"file://{output_file}"
                    url_path = full_url

                log.info(f"文档已保存到: {output_file}")

                # 使用完整URL作为访问链接，包含域名和端口
                save_message = (
                    f"\n\n文档已生成并保存。\n访问链接: [WMS需求文档]({full_url})"
                )
            except Exception as e:
                log.error(f"保存文档失败: {e}")
                save_message = "\n\n注意: 文档已生成但保存失败。"

            # 生成文档摘要作为响应
            response = (
                f"## WMS系统用户需求说明生成完成\n\n"
                f"**文档ID**: {document_data.get('doc_metadata', {}).get('document_id', 'URS-WMS-001')}\n"
                f"**版本**: {document_data.get('doc_metadata', {}).get('version', '1.0')}\n"
                f"**日期**: {document_data.get('doc_metadata', {}).get('date', datetime.now().strftime('%Y-%m-%d'))}\n\n"
                f"**文档目的**: {document_data.get('purpose', '')}\n\n"
                f"**需求项总数**: {len(document_data.get('requirements', []))} 项\n"
                f"- 功能性需求: {len([r for r in document_data.get('requirements', []) if r.get('category') == '功能性需求'])} 项\n"
                f"- 非功能性需求: {len([r for r in document_data.get('requirements', []) if r.get('category') == '非功能性需求'])} 项\n"
                f"- 控制需求: {len([r for r in document_data.get('requirements', []) if r.get('category') == '控制需求'])} 项"
                f"{save_message}"
            )

            return state, response

        except json.JSONDecodeError as e:
            log.error(f"JSON解析错误: {e}", exc_info=True)
            state["stage"] = "feedback"
            return (
                state,
                f"在生成文档时出错: JSON格式错误 - {e}\n\n请重新尝试或提供更多需求。",
            )
        except Exception as e:
            log.error(f"生成文档错误: {e}", exc_info=True)
            state["stage"] = "feedback"
            return state, f"在生成文档时出错: {e}\n\n请重新尝试或提供更多需求。"

    async def _continue_analysis(
        self, state: WMSRequirementsState, user_input: str, emitter: Optional[Callable]
    ) -> Tuple[str, WMSRequirementsState]:
        """继续需求分析流程

        根据当前状态和用户新输入继续需求分析流程。

        参数：
            state: 当前状态
            user_input: 用户输入的需求描述
            emitter: 事件发射器函数

        返回：
            Tuple[str, WMSRequirementsState]: 响应内容和更新后的状态
        """
        log.info(f"继续需求分析，当前阶段: {state['stage']}")
        await self.emit_status(emitter, "info", "处理新的需求输入...")

        try:
            # 根据当前阶段决定下一步操作
            current_stage = state["stage"]

            if current_stage == "input" or current_stage == "feedback":
                # 重新分析需求
                log.debug("重新分析需求")
                # 合并用户输入
                merged_input = (
                    "\n".join(
                        [
                            msg["content"]
                            for msg in state["messages"]
                            if msg["role"] == "user"
                        ]
                    )
                    + "\n"
                    + user_input
                )

                # 执行分析
                return await self._analyze_requirements(state, merged_input, emitter)

            elif current_stage == "rag_query":
                # 已经分析需求，正在进行RAG查询
                log.debug("补充需求后继续 RAG 查询")
                # 更新需求点
                await self.emit_status(emitter, "info", "更新需求点...")
                updated_state = await self._generate_requirement_points(
                    state, user_input, emitter
                )

                # 执行RAG查询
                updated_state, response = await self._perform_rag_query(
                    updated_state, emitter
                )
                return response, updated_state

            elif current_stage == "document_writer":
                # 正在生成文档
                log.debug("补充需求后重新生成文档")
                await self.emit_status(emitter, "info", "基于补充信息重新生成文档...")

                # 重新生成文档
                updated_state, response = await self._generate_document(state, emitter)
                return response, updated_state

            elif current_stage == "complete":
                # 分析已完成，用户可能希望修改某些内容
                log.debug("分析已完成，处理用户请求")
                await self.emit_status(emitter, "info", "分析用户反馈...")

                # 准备LLM输入
                messages = [
                    self.sys_msg,
                    HumanMessage(
                        content=f"""分析此用户反馈，并确定是否需要对WMS需求文档进行修改。

当前状态: WMS需求文档已生成完成。

用户反馈:
{user_input}

任务:
1. 分析用户反馈是否要求对文档进行修改
2. 确定应该采取的行动: "no_change"(无需修改) 或 "regenerate"(重新生成)
3. 提供简短的解释

输出格式: JSON对象，格式如下:
{{
  "action": "no_change或regenerate",
  "explanation": "解释原因"
}}
"""
                    ),
                ]

                # 调用LLM分析反馈
                ai_response = await self.llm.ainvoke(messages)
                log.debug(f"LLM响应: {ai_response.content}")

                # 解析响应
                try:
                    content = ai_response.content
                    # 提取JSON
                    if "```json" in content and "```" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                    elif "{" in content and "}" in content:
                        json_str = content[content.find("{") : content.rfind("}") + 1]
                    else:
                        json_str = content

                    analysis_result = json.loads(json_str)
                    action = analysis_result.get("action", "no_change")
                    explanation = analysis_result.get("explanation", "")

                    if action == "regenerate":
                        # 重新生成文档
                        log.info("基于用户反馈重新生成文档")
                        await self.emit_status(
                            emitter, "info", "根据您的反馈重新生成文档..."
                        )

                        # 重置状态为RAG查询阶段
                        state["stage"] = "rag_query"

                        # 重新生成需求点
                        merged_input = (
                            "\n".join(
                                [
                                    msg["content"]
                                    for msg in state["messages"]
                                    if msg["role"] == "user"
                                ]
                            )
                            + "\n"
                            + user_input
                        )

                        updated_state = await self._generate_requirement_points(
                            state, merged_input, emitter
                        )
                        updated_state, rag_response = await self._perform_rag_query(
                            updated_state, emitter
                        )

                        return (
                            f"理解您的反馈: {explanation}\n\n{rag_response}",
                            updated_state,
                        )
                    else:
                        # 无需更改
                        return (
                            f"感谢您的反馈。{explanation}\n\n如果需要对文档进行其他修改，请提供更多详细信息。",
                            state,
                        )

                except Exception as e:
                    log.error(f"解析用户反馈错误: {e}", exc_info=True)
                    return (
                        f"感谢您的反馈。我已记录您的意见，但处理过程中出现了错误。如果需要修改文档，请提供更具体的说明。",
                        state,
                    )

            else:
                # 未知状态，重置为输入阶段
                log.warning(f"未知状态: {current_stage}，重置为输入阶段")
                state["stage"] = "input"
                return await self._analyze_requirements(state, user_input, emitter)

        except Exception as e:
            log.error(f"继续分析错误: {e}", exc_info=True)
            error_response = f"对不起，在处理您的输入时发生错误: {e}\n\n请重新尝试或提供不同的需求描述。"
            return error_response, state
