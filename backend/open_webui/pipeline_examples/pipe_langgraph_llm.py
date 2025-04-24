"""
title: LangGraph LLM Pipe
author: OpenWebUI Developer
version: 0.1.0
license: MIT
requirements: langgraph>=0.0.15, langchain-openai>=0.0.1, pydantic>=2.11.0
environment_variables: [AZURE_API_KEY, AZURE_ENDPOINT, AZURE_API_VERSION, AZURE_DEPLOYMENT_NAME]
Supports: [Basic Processing, Streaming, LangGraph Integration]
"""

import os
import logging
import asyncio
import time
from typing import List, Dict, Union, AsyncGenerator, Optional, Callable, Any
from pydantic import BaseModel, Field

# LangGraph 和 LangChain 导入
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import AzureChatOpenAI

# 配置日志记录
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class State(TypedDict):
    """LangGraph 状态类型定义"""
    messages: Annotated[list, add_messages]

class Pipe:
    """
    基于 LangGraph 的 LLM 管道，使用 Azure OpenAI 服务
    """
    # 核心属性
    type = "pipe"
    id = "langgraph_llm"
    name = "LangGraph LLM Pipe"

    # 配置类 (Valves)
    class Valves(BaseModel):
        AZURE_API_KEY: Optional[str] = Field(
            default="AZURE_API_KEY", 
            description="Azure OpenAI API Key"
        )
        AZURE_ENDPOINT: Optional[str] = Field(
            default="AZURE_ENDPOINT",
            description="Azure OpenAI Endpoint URL"
        )
        AZURE_API_VERSION: str = Field(
            default=os.getenv("AZURE_API_VERSION", "2023-09-01-preview"),
            description="Azure OpenAI API Version"
        )
        AZURE_DEPLOYMENT_NAME: str = Field(
            default=os.getenv("AZURE_DEPLOYMENT_NAME", "NNITAsia-GPT-4o"),
            description="Azure OpenAI Deployment Name"
        )
        emit_interval: float = Field(
            default=0.5, 
            ge=0, 
            description="Status emit interval (s)"
        )
        enable_status_indicator: bool = Field(
            default=True, 
            description="Enable status emits"
        )
        DEBUG: bool = Field(
            default=False, 
            description="Enable debug logging"
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
        """初始化 LangGraph LLM Pipe"""
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.graph = None
        self.llm = None
        
        log.info(f"Pipe '{self.name}' (ID: {self.id}) initialized.")
        
        if self.valves.DEBUG:
            log.setLevel(logging.DEBUG)
            log.debug(f"Pipe '{self.id}' running in DEBUG mode.")
        
        # 配置验证
        if not self.valves.AZURE_API_KEY:
            log.warning(f"Pipe '{self.id}': AZURE_API_KEY environment variable not set.")
        
        if not self.valves.AZURE_ENDPOINT:
            log.warning(f"Pipe '{self.id}': AZURE_ENDPOINT environment variable not set.")

    def _init_langgraph(self):
        """初始化 LangGraph 组件"""
        try:
            # 检查必要的配置
            if not self.valves.AZURE_API_KEY or not self.valves.AZURE_ENDPOINT:
                log.error("Cannot initialize LangGraph: Missing API Key or Endpoint")
                return False
                
            log.info(f"Initializing LangGraph with Azure OpenAI deployment: {self.valves.AZURE_DEPLOYMENT_NAME}")
            
            # 初始化 LLM
            self.llm = AzureChatOpenAI(
                openai_api_key=self.valves.AZURE_API_KEY,
                azure_endpoint=self.valves.AZURE_ENDPOINT,
                deployment_name=self.valves.AZURE_DEPLOYMENT_NAME,
                openai_api_version=self.valves.AZURE_API_VERSION,
            )
            
            # 测试 LLM 连接
            log.debug("Testing LLM connection...")
            
            # 构建 LangGraph
            log.debug("Building LangGraph...")
            graph_builder = StateGraph(State)
            
            # 添加 chatbot 节点
            graph_builder.add_node("chatbot", self._chatbot)
            graph_builder.add_edge(START, "chatbot")
            graph_builder.add_edge("chatbot", END)
            
            # 编译图
            log.debug("Compiling LangGraph...")
            self.graph = graph_builder.compile()
            log.info(f"LangGraph successfully initialized for '{self.id}'")
            return True
            
        except Exception as e:
            log.error(f"Failed to initialize LangGraph: {e}", exc_info=True)
            self.graph = None
            self.llm = None
            return False

    def _chatbot(self, state: State) -> Dict:
        """LangGraph chatbot 节点函数"""
        try:
            if not self.llm:
                log.error("LLM not initialized in chatbot node")
                raise ValueError("LLM not initialized")
                
            log.debug(f"Processing chatbot node with messages count: {len(state['messages'])}")
            # 安全检查消息格式
            if not state["messages"] or not isinstance(state["messages"], list):
                log.error(f"Invalid messages format: {state['messages']}")
                raise ValueError("Invalid messages format")
                
            result = {"messages": [self.llm.invoke(state["messages"])]}
            log.debug(f"Chatbot node completed successfully")
            return result
        except Exception as e:
            log.error(f"Error in chatbot node: {e}", exc_info=True)
            raise

    def pipes(self) -> List[dict]:
        """返回此 Pipe 提供的模型/功能列表"""
        return [
            {
                "id": f"{self.id}/azure",
                "name": f"{self.name} (Azure)",
                "context_length": 128000,  # GPT-4o context window
                "supports_vision": True,   # GPT-4o supports vision
                "supports_functions": True, # GPT-4o supports function calling
                "supports_stream": True    # 支持流式响应
            }
        ]

    async def pipe(self, body: Dict, **kwargs) -> Union[Dict, AsyncGenerator[Union[str, Dict], None]]:
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
        
        log.debug(f"Pipe '{self.id}' received request. Body: {body}, User: {user}")
        await self.emit_status(emitter, "info", "Processing started...")
        
        try:
            # 验证配置
            if not self.valves.AZURE_API_KEY or not self.valves.AZURE_ENDPOINT:
                error_msg = "Configuration Error: Azure API Key or Endpoint is missing"
                log.error(error_msg)
                await self.emit_status(emitter, "error", error_msg, True)
                return {"error": error_msg}
            
            # 初始化 LangGraph（如果尚未初始化）
            if not self.graph or not self.llm:
                log.info("LangGraph not initialized, initializing now...")
                await self.emit_status(emitter, "info", "Initializing LangGraph...")
                init_success = self._init_langgraph()
                
                if not init_success:
                    error_msg = "Failed to initialize LangGraph: Missing credentials or connection error"
                    log.error(error_msg)
                    await self.emit_status(emitter, "error", error_msg, True)
                    return {"error": error_msg}
                
            if not self.graph:
                error_msg = "Failed to initialize LangGraph: Unknown error"
                log.error(error_msg)
                await self.emit_status(emitter, "error", error_msg, True)
                return {"error": error_msg}
            
            # 处理输入
            messages = self._process_messages(body.get("messages", []))
            
            # 根据 Valves 配置决定是否启用流式响应
            is_stream = body.get("stream", False)
            if self.valves.FORCE_STREAMING is not None:
                original_stream = is_stream
                is_stream = self.valves.FORCE_STREAMING
                log.info(f"流式响应设置已被 Valves 配置覆盖: {original_stream} -> {is_stream}")
            
            log.debug(f"Processing with stream={is_stream}, message count={len(messages)}")
            
            # 准备 LangGraph 输入
            graph_input = {"messages": messages}
            
            if is_stream:
                # 返回异步生成器
                log.info("Starting stream response...")
                return self._stream_graph_updates(graph_input, emitter)
            else:
                # 非流式响应
                log.info("Generating non-stream response...")
                await self.emit_status(emitter, "info", "Calling LangGraph...")
                
                # 执行图并获取结果
                result = self.graph.invoke(graph_input)
                response_content = result["messages"][-1].content
                
                log.info(f"Generated non-stream response: {len(response_content)} chars")
                log.debug(f"Response content: {response_content[:100]}...")
                await self.emit_status(emitter, "info", "Processing complete.", True)
                
                # 注意：非流式响应必须是字典格式，并包含 content 和 format 字段
                return {"content": response_content, "format": "text"}
                
        except asyncio.TimeoutError:
            log.error(f"Request timed out", exc_info=True)
            await self.emit_status(emitter, "error", "Request timed out", True)
            return {"error": "Request timed out"}
        except Exception as e:
            log.error(f"Unhandled error: {e}", exc_info=True)
            await self.emit_status(emitter, "error", f"An unexpected error occurred: {e}", True)
            return {"error": f"An unexpected error occurred: {e}"}

    async def _stream_graph_updates(self, graph_input: Dict, emitter: Optional[Callable]) -> AsyncGenerator[Dict, None]:
        """
        流式处理 LangGraph 更新
        
        Args:
            graph_input: LangGraph 输入
            emitter: 事件发射器
            
        Yields:
            流式响应块
        """
        try:
            await self.emit_status(emitter, "info", "Streaming started...")
            log.info("Starting stream response generation")
            
            # 创建一个队列来存储流式结果
            queue = asyncio.Queue()
            
            # 在线程中运行 graph.stream 并将结果放入队列
            async def stream_in_thread():
                try:
                    log.debug(f"Starting LangGraph stream with input: {graph_input}")
                    for event in self.graph.stream(graph_input):
                        log.debug(f"Received event from LangGraph: {event}")
                        for key, value in event.items():
                            log.debug(f"Processing event key: {key}")
                            if "messages" in value and value["messages"]:
                                content = value["messages"][-1].content
                                log.debug(f"Extracted content: {content[:50]}...")
                                # 根据配置的块大小拆分内容
                                if self.valves.STREAM_CHUNK_SIZE > 0:
                                    # 将内容按照指定大小分块
                                    chunk_size = self.valves.STREAM_CHUNK_SIZE
                                    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
                                    log.debug(f"内容被分成 {len(chunks)} 个块进行流式传输")
                                    
                                    for chunk in chunks:
                                        # 注意这里使用字符串而不是字典，符合 OpenWebUI 的流式响应格式
                                        await queue.put(chunk)
                                        # 添加小延迟以模拟真实的流式效果
                                        await asyncio.sleep(0.05)
                                else:
                                    # 不分块，直接发送完整内容
                                    await queue.put(content)
                    # 标记流结束
                    log.debug("Stream complete, marking end")
                    await queue.put(None)
                except Exception as e:
                    log.error(f"Error in stream thread: {e}", exc_info=True)
                    await queue.put({"type": "error", "content": f"Stream error: {e}"})
                    await queue.put(None)
            
            # 启动流处理任务
            task = asyncio.create_task(stream_in_thread())
            
            # 从队列中获取结果并产生
            while True:
                chunk = await queue.get()
                if chunk is None:
                    # 流结束
                    log.debug("End of stream reached")
                    break
                elif isinstance(chunk, dict) and chunk.get("type") == "error":
                    # 错误情况
                    log.error(f"Error in stream: {chunk['content']}")
                    yield chunk  # 错误保持字典格式
                    await self.emit_status(emitter, "error", chunk["content"], True)
                    break
                else:
                    # 正常块 - 直接返回字符串，符合 OpenWebUI 流式格式
                    log.debug(f"Yielding chunk: {chunk[:50]}...")
                    yield chunk  # 直接返回字符串
            
            # 等待任务完成
            await task
            await self.emit_status(emitter, "info", "Stream complete.", True)
            log.info("Stream processing completed successfully")
            
        except Exception as e:
            log.error(f"Error during streaming: {e}", exc_info=True)
            yield {"type": "error", "content": f"Stream error: {e}"}
            await self.emit_status(emitter, "error", f"Stream error: {e}", True)

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
            log.debug(f"Status emission skipped (emitter: {bool(emitter)}, enabled: {self.valves.enable_status_indicator})")
            return
            
        current_time = time.time()
        # 除非是最终的 'done' 消息，否则限制发送频率
        if not done and (current_time - self.last_emit_time < self.valves.emit_interval):
            log.debug(f"Status emission throttled: '{message}'")
            return
            
        try:
            status_payload = {
                "type": "status", 
                "level": level,
                "message": message,
                "done": done
            }
            log.debug(f"Emitting status: {status_payload}")
            await emitter(status_payload)
            self.last_emit_time = current_time
        except Exception as e:
            # 记录错误但不中断管道
            log.error(f"Failed to emit status (level='{level}', msg='{message}'): {e}")

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
                processed.append({"role": role, "content": content})
            else:
                log.warning(f"Skipping invalid message format: {msg}")
                
        log.debug(f"Processed messages ({len(processed)} total): {processed if self.valves.DEBUG else '...'}")
        return processed

# # 不需要创建实例，直接使用 Pipe 类
