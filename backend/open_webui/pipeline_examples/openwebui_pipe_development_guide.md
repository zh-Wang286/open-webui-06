# OpenWebUI Pipe 开发规范指南 (精简版)

## 1. 框架概述

OpenWebUI Pipe 允许通过 Python 类扩展功能，集成 LLM、API、工具协议 (MCP) 或 LangChain 等。核心是模块化和可配置性。

## 2. Pipe 基础语法规范

### 2.1 基础类结构

```python
import os
from pydantic import BaseModel, Field
from typing import List, Dict, Union, AsyncGenerator, Optional, Callable
import time
import asyncio
import logging

log = logging.getLogger(__name__)
# Basic logging config, should be configured externally in a real app
logging.basicConfig(level=logging.INFO)

class Pipe:
    # 2.1.1 配置类 (Valves)
    class Valves(BaseModel):
        API_KEY: Optional[str] = Field(default=os.getenv("MY_API_KEY"), description="API Key")
        TIMEOUT: int = Field(default=30, description="Request timeout")
        emit_interval: float = Field(default=0.5, ge=0, description="Status emit interval (s)")
        enable_status_indicator: bool = Field(default=True, description="Enable status emits")
        DEBUG: bool = Field(default=False, description="Enable debug logging")

    # 2.1.2 核心属性
    type = "pipe"
    id = "my_pipe"         # 唯一ID (snake_case)
    name = "My Custom Pipe" # 显示名称

    # 2.1.3 初始化 (__init__)
    def __init__(self):
        self.valves = self.Valves()
        self.last_emit_time = 0
        log.info(f"Pipe '{self.name}' (ID: {self.id}) initialized.")
        if self.valves.DEBUG:
            log.setLevel(logging.DEBUG)
            log.debug(f"Pipe '{self.id}' running in DEBUG mode.")
        # 配置验证
        if not self.valves.API_KEY:
            log.warning(f"Pipe '{self.id}': MY_API_KEY environment variable not set.")
            # Consider raising an error if critical: 
            # raise ValueError(f"Pipe '{self.id}': MY_API_KEY is required but not set.")

    # 2.3.1 'pipes' 方法 (必需)
    def pipes(self) -> List[dict]:
        # 返回此 Pipe 提供的模型/功能列表
        # Example: Define models provided by this pipe
        return [
            {
                "id": f"{self.id}/default", # Unique ID for the model/variant
                "name": self.name,           # Display name in UI
                "context_length": 4096,    # Optional: Model context window size
                "supports_vision": False,  # Optional: Vision capability
                "supports_functions": False # Optional: Function calling capability
            },
            # Add more variants if needed
        ]

    # 2.3.2 'pipe' 方法 (必需, 异步)
    async def pipe(self, body: Dict, **kwargs) -> Union[str, Dict, AsyncGenerator[Union[str, Dict], None]]:
        emitter = kwargs.get("__event_emitter__")
        user = kwargs.get("__user__") # Example of accessing user info if passed
        
        log.debug(f"Pipe '{self.id}' received request. Body: {body}, User: {user}")
        await self.emit_status(emitter, "info", "Processing started...")

        try:
            # 1. 验证配置 (Example)
            if not self.valves.API_KEY:
                 # Log already warned in __init__, decide if fatal here
                 await self.emit_status(emitter, "error", "Configuration Error: API Key is missing", True)
                 return {"error": "Configuration Error: API Key is missing"}

            # 2. 处理输入
            messages = self._process_messages(body.get("messages", []))
            model_id = body.get("model") # The specific model/variant requested
            is_stream = body.get("stream", False)
            log.debug(f"Processing for model: {model_id}, stream: {is_stream}")

            # 3. 调用后端服务 (模拟)
            await self.emit_status(emitter, "info", "Calling backend service...")
            await asyncio.sleep(0.5) # Simulate async work

            if is_stream:
                # 返回异步生成器
                log.debug("Starting stream response...")
                async def stream_response():
                    try:
                        yield {"type": "chunk", "content": "Streaming response started... "}
                        await asyncio.sleep(0.2)
                        yield {"type": "chunk", "content": "More content... "}
                        await asyncio.sleep(0.2)
                        yield {"type": "final_chunk", "content": "Stream ended."}
                        log.debug("Stream finished successfully.")
                        await self.emit_status(emitter, "info", "Stream complete.", True)
                    except Exception as stream_err:
                        log.error(f"Error during streaming: {stream_err}", exc_info=True)
                        # Attempt to yield error message if possible
                        yield {"type": "error", "content": f"Stream error: {stream_err}"}
                        await self.emit_status(emitter, "error", f"Stream error: {stream_err}", True)
                return stream_response()
            else:
                # 非流式响应
                log.debug("Generating non-stream response...")
                # response_data = await self._call_backend(messages, model_id)
                response_data = {"content": f"Completed response for {model_id}", "format": "text"}
                log.debug(f"Generated non-stream response: {response_data}")
                await self.emit_status(emitter, "info", "Processing complete.", True)
                return response_data

        except asyncio.TimeoutError:
            log.error(f"Pipe '{self.id}' timed out after {self.valves.TIMEOUT}s", exc_info=True)
            await self.emit_status(emitter, "error", "Request timed out", True)
            return {"error": "Request timed out"}
        except Exception as e:
            log.error(f"Unhandled error in pipe '{self.id}': {e}", exc_info=True)
            await self.emit_status(emitter, "error", f"An unexpected error occurred: {e}", True)
            # Return a generic error message to the user
            return {"error": "An unexpected error occurred. Please check logs."}

    # 2.4 事件发射器辅助方法
    async def emit_status(self, emitter: Optional[Callable], level: str, message: str, done: bool = False):
        if not emitter or not self.valves.enable_status_indicator:
            log.debug(f"Status emission skipped (emitter: {bool(emitter)}, enabled: {self.valves.enable_status_indicator})")
            return
            
        current_time = time.time()
        # Throttle emissions unless it's the final 'done' message
        if not done and (current_time - self.last_emit_time < self.valves.emit_interval):
            log.debug(f"Status emission throttled: '{message}'")
            return
            
        try:
            status_payload = {
                "type": "status", 
                "level": level, # 'info', 'warning', 'error', 'critical'
                "message": message,
                "done": done
            }
            log.debug(f"Emitting status: {status_payload}")
            await emitter(status_payload)
            self.last_emit_time = current_time
        except Exception as e:
            # Log error but don't crash the pipe
            log.error(f"Failed to emit status (level='{level}', msg='{message}'): {e}")

    # 6. 消息处理辅助方法
    def _process_messages(self, messages: List[Dict]) -> List[Dict]:
        # Example: Convert OpenWebUI message format to a backend-specific format
        # This is highly dependent on the target service (LLM API, etc.)
        processed = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")
            # Simple validation/transformation
            if role and content:
                processed.append({"role": role, "content": content})
            else:
                log.warning(f"Skipping invalid message format: {msg}")
        log.debug(f"Processed messages ({len(processed)} total): {processed if self.valves.DEBUG else '...'}") # Avoid logging potentially large content unless DEBUG
        return processed

# 2.2 模块元数据 (模块级文档字符串)
"""
title: My Custom Pipe (Concise Example)
author: Your Name
version: 0.1.1
license: MIT
requirements: pydantic>=2.0.0, aiohttp # List dependencies
environment_variables: [MY_API_KEY] # List required env vars
Supports: [Basic Processing, Streaming (Example)] # List features
"""

### 2.5 消息格式

-   **输入**: `body['messages']` 是 `[{'role': 'user', 'content': '...'}, ...]` 列表。
-   **输出**:
    -   非流式: 推荐 `{"content": "...", "format": "text"}` 或 `{"error": "..."}`。
    -   流式: `AsyncGenerator` yield 字符串或字典 (推荐字典 `{"type": "chunk", "content": "..."}` 或 `{"type": "error", ...}` 来区分)。

## 3. 配置管理规范 (`Valves`)

-   使用内部类 `Valves(BaseModel)` 定义配置。
-   用 `Field` 设置默认值、环境变量 (`os.getenv`)、描述和验证规则 (e.g., `ge=0`)。
-   在 `__init__` 中实例化 `self.valves = self.Valves()`。
-   在代码中通过 `self.valves.CONFIG_NAME` 访问。

## 4. 工具集成规范

-   封装外部调用（API、MCP、LangChain）到独立的异步辅助方法中。
-   通过 `Valves` 安全管理凭据和端点。
-   提供清晰的错误处理和状态反馈 (通过 `emit_status` 和日志)。

## 5. 异步处理规范

-   **强制**: `pipe` 和所有 I/O 操作（网络、文件、长时间计算）必须是 `async def`。
-   使用 `await` 调用其他协程。
-   流式响应使用 `AsyncGenerator`。
-   使用支持异步的库 (e.g., `aiohttp`, `asyncio`)。

## 6. 消息构建规范

-   实现 `_process_messages` 或类似方法，将 OpenWebUI 的消息列表转换成目标服务期望的格式。
-   处理系统提示、多模态内容（如果支持）。
-   考虑上下文长度限制，可能需要截断。

## 7. 扩展开发规范

1.  创建新的 `.py` 文件 (e.g., `my_pipe.py`) 在 OpenWebUI 的 `pipelines` 目录或子目录。
2.  定义 `MyPipe` 类结构 (参照 2.1)。
3.  添加模块级文档字符串元数据 (参照 2.2)。
4.  实现 `Valves`, `__init__`, `pipes`, `async pipe`。
5.  编写必要的异步辅助方法 (e.g., `_call_backend`, `_process_messages`)。
6.  确保所有依赖项在 `requirements` 元数据中列出。

## 8. 调试与测试

### 8.1 日志记录

-   使用标准 `logging` 模块 (`log = logging.getLogger(__name__)`)。
-   合理使用日志级别 (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`)。
-   通过 `Valves.DEBUG` 配置项控制 `DEBUG` 级别日志的输出。
-   记录关键事件、配置值（**不要记录明文 API Key**）、输入/输出摘要、错误。

### 8.2 错误处理

-   在 `pipe` 方法顶层和所有可能失败的 I/O 操作处使用 `try...except` 块。
-   捕获具体的异常类型。
-   记录详细错误信息和堆栈跟踪 (`log.error(..., exc_info=True)`)。
-   通过 `emit_status` 向前端报告用户友好的错误 (`level="error"`, `done=True`)。
-   向调用者返回明确的错误响应 `{"error": "..."}`。

### 8.3 测试

-   **单元测试**: 使用 `pytest` 和 `pytest-asyncio` 测试辅助方法。使用 `unittest.mock` 模拟依赖。
-   **集成测试**: 测试 `pipe` 方法的整体流程。
    -   构造测试用的 `body` 字典。
    -   使用 `unittest.mock.AsyncMock` 模拟 `__event_emitter__` 并断言其调用。
    -   使用 `aioresponses` 或类似库模拟外部 HTTP API 响应。
    -   覆盖成功路径（流式和非流式）以及各种错误情况（配置错误、API 错误、超时）。
