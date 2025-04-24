"""
title: LangGraph Arithmetic Assistant Pipe
author: OpenWebUI Developer
version: 0.1.0
license: MIT
requirements: langgraph>=0.0.15, langchain-openai>=0.0.1, pydantic>=2.11.0
environment_variables: [OPENAI_API_KEY, AZURE_ENDPOINT, OPENAI_API_VERSION, MODEL_NAME]
Supports: [Basic Processing, Streaming, LangGraph Integration, Tool Calling]
"""

import os
import logging
import asyncio
import time
from typing import List, Dict, Union, AsyncGenerator, Optional, Callable, Any
from pydantic import BaseModel, Field

# LangGraph 和 LangChain 导入
from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_openai import AzureChatOpenAI

# 配置日志记录
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Pipe:
    """
    基于 LangGraph 的算术助手管道，使用 Azure OpenAI 服务
    """
    # 核心属性
    type = "pipe"
    id = "arithmetic_assistant"
    name = "LangGraph Arithmetic Assistant"

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
        """初始化 LangGraph 算术助手 Pipe"""
        self.valves = self.Valves()
        self.last_emit_time = 0
        self.graph = None
        self.llm = None
        self.llm_with_tools = None
        self.sys_msg = None
        
        log.info(f"Pipe '{self.name}' (ID: {self.id}) initialized.")
        
        if self.valves.DEBUG:
            log.setLevel(logging.DEBUG)
            log.debug(f"Pipe '{self.id}' running in DEBUG mode.")
        
        # 配置验证
        if not self.valves.OPENAI_API_KEY:
            log.warning(f"Pipe '{self.id}': OPENAI_API_KEY environment variable not set.")
        
        if not self.valves.AZURE_ENDPOINT:
            log.warning(f"Pipe '{self.id}': AZURE_ENDPOINT environment variable not set.")
        
        # 定义工具
        self.tools = [self.add, self.multiply, self.divide]
        log.info(f"Tools registered: {[tool.__name__ for tool in self.tools]}")
        
        # 系统消息
        self.sys_msg = SystemMessage(
            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        )
        log.info("System message configured")

    # ======================
    # ARITHMETIC TOOLS
    # ======================
    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiply a and b."""
        result = a * b
        log.info(f"Executing multiply({a}, {b}) = {result}")
        return result

    @staticmethod
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        result = a + b
        log.info(f"Executing add({a}, {b}) = {result}")
        return result

    @staticmethod
    def divide(a: int, b: int) -> float:
        """Divide a and b."""
        result = a / b
        log.info(f"Executing divide({a}, {b}) = {result}")
        return result

    def _init_langgraph(self):
        """初始化 LangGraph 组件"""
        try:
            # 检查必要的配置
            if not self.valves.OPENAI_API_KEY or not self.valves.AZURE_ENDPOINT:
                log.error("Cannot initialize LangGraph: Missing API Key or Endpoint")
                return False
                
            log.info(f"Initializing LangGraph with Azure OpenAI deployment: {self.valves.MODEL_NAME}")
            
            # 初始化 LLM
            self.llm = AzureChatOpenAI(
                openai_api_key=self.valves.OPENAI_API_KEY,
                azure_endpoint=self.valves.AZURE_ENDPOINT,
                deployment_name=self.valves.MODEL_NAME,
                openai_api_version=self.valves.OPENAI_API_VERSION,
            )
            
            # 绑定工具到LLM
            self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
            log.info("LLM initialized with tools")
            
            # 构建图
            builder = StateGraph(MessagesState)
            
            # 定义节点
            builder.add_node("assistant", self._assistant_node)
            builder.add_node("tools", ToolNode(self.tools))
            log.info("Nodes added to graph")
            
            # 定义边
            builder.add_edge(START, "assistant")
            builder.add_conditional_edges(
                "assistant",
                tools_condition,
                {"tools": "tools", "__end__": "__end__"}
            )
            builder.add_edge("tools", "assistant")
            log.info("Edges configured in graph")
            
            # 编译图
            self.graph = builder.compile()
            log.info("Graph compilation complete")
            return True
            
        except Exception as e:
            log.error(f"Failed to initialize LangGraph: {e}", exc_info=True)
            return False

    def _assistant_node(self, state: MessagesState):
        """处理助手节点"""
        log.info("Processing assistant node...")
        messages = [self.sys_msg] + state["messages"]
        log.debug(f"Input messages to LLM: {messages}")
        
        response = self.llm_with_tools.invoke(messages)
        log.debug(f"LLM response: {response}")
        
        return {"messages": [response]}

    def pipes(self) -> List[dict]:
        """返回此 Pipe 提供的模型/功能列表"""
        return [
            {
                "id": f"{self.id}/default",
                "name": self.name,
                "context_length": 4096,
                "supports_vision": False,
                "supports_functions": True
            }
        ]

    async def pipe(self, body: Dict, **kwargs) -> Union[str, Dict, AsyncGenerator[Union[str, Dict], None]]:
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
            # 1. 验证配置
            if not self.valves.OPENAI_API_KEY or not self.valves.AZURE_ENDPOINT:
                await self.emit_status(emitter, "error", "Configuration Error: API Key or Endpoint is missing", True)
                return {"error": "Configuration Error: API Key or Endpoint is missing"}

            # 2. 处理输入
            messages = self._process_messages(body.get("messages", []))
            model_id = body.get("model", f"{self.id}/default")
            is_stream = self.valves.FORCE_STREAMING if self.valves.FORCE_STREAMING is not None else body.get("stream", False)
            log.debug(f"Processing for model: {model_id}, stream: {is_stream}")

            # 3. 初始化 LangGraph (如果尚未初始化)
            if self.graph is None:
                await self.emit_status(emitter, "info", "Initializing LangGraph...")
                if not self._init_langgraph():
                    await self.emit_status(emitter, "error", "Failed to initialize LangGraph", True)
                    return {"error": "Failed to initialize LangGraph"}

            # 4. 准备输入
            graph_input = {"messages": [HumanMessage(content=msg["content"]) for msg in messages if msg["role"] == "user"]}
            if not graph_input["messages"]:
                await self.emit_status(emitter, "error", "No valid user messages found", True)
                return {"error": "No valid user messages found"}

            # 5. 处理请求
            if is_stream:
                # 流式响应
                await self.emit_status(emitter, "info", "Starting stream processing...")
                return self._stream_graph_updates(graph_input, emitter)
            else:
                # 非流式响应
                await self.emit_status(emitter, "info", "Processing with LangGraph...")
                try:
                    result = self.graph.invoke(graph_input)
                    final_message = result["messages"][-1].content
                    log.info(f"Final result: {final_message}")
                    await self.emit_status(emitter, "info", "Processing complete.", True)
                    return {"content": final_message, "format": "text"}
                except Exception as e:
                    log.error(f"Error during graph processing: {e}", exc_info=True)
                    await self.emit_status(emitter, "error", f"Processing error: {e}", True)
                    return {"error": f"Processing error: {e}"}

        except asyncio.TimeoutError:
            log.error(f"Pipe '{self.id}' timed out", exc_info=True)
            await self.emit_status(emitter, "error", "Request timed out", True)
            return {"error": "Request timed out"}
        except Exception as e:
            log.error(f"Unexpected error in pipe: {e}", exc_info=True)
            await self.emit_status(emitter, "error", f"An unexpected error occurred: {e}", True)
            return {"error": "An unexpected error occurred. Please check logs."}

    async def _stream_graph_updates(self, graph_input: Dict, emitter: Optional[Callable]) -> AsyncGenerator[Union[str, Dict], None]:
        """
        流式处理 LangGraph 更新
        
        Args:
            graph_input: LangGraph 输入
            emitter: 事件发射器
            
        Yields:
            流式响应块
        """
        try:
            # 创建队列用于线程间通信
            queue = asyncio.Queue()
            
            # 在线程中运行图并将结果放入队列
            async def stream_in_thread():
                try:
                    log.debug("Starting graph execution in thread")
                    # 执行图并获取结果
                    result = self.graph.invoke(graph_input)
                    final_message = result["messages"][-1].content
                    log.debug(f"Graph execution complete, final message: {final_message}")
                    
                    # 模拟流式传输 - 将最终消息分块
                    chunk_size = self.valves.STREAM_CHUNK_SIZE
                    for i in range(0, len(final_message), chunk_size):
                        chunk = final_message[i:i+chunk_size]
                        log.debug(f"Queuing chunk: {chunk}")
                        await queue.put(chunk)
                        # 添加小延迟以模拟真实流式传输
                        await asyncio.sleep(0.05)
                    
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

# 不需要创建实例，直接使用 Pipe 类