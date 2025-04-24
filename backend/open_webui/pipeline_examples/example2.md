# MCP Pipe 框架分析与使用指南

## 框架概述

MCP Pipe 是一个用于在 Open WebUI 中集成 MCP (Modular Command Protocol) 服务器的功能管道框架。它允许开发者通过配置多个 MCP 服务器，并将它们的工具和提示功能集成到聊天界面中。

## 核心组件分析

### 1. MCPClient 类

`MCPClient` 是与 MCP 服务器交互的核心类，主要功能包括：

- **多服务器连接**：通过 `mcp_config.json` 配置连接多个 MCP 服务器
- **工具管理**：自动发现和调用各服务器提供的工具
- **提示管理**：支持从服务器获取预定义的提示模板
- **会话管理**：维护与各服务器的连接会话

#### 关键方法：
- `connect_to_servers()`: 根据配置连接所有服务器
- `call_tool()`: 调用指定工具
- `get_prompt()`: 获取服务器上的提示模板
- `process_query()`: 处理用户查询，协调工具调用

### 2. Pipe 类

`Pipe` 类是 Open WebUI 的接口层，主要功能包括：

- **配置管理**：通过 `Valves` 子类管理各种参数
- **消息构建**：构建包含工具信息的系统消息
- **事件处理**：处理 Open WebUI 的事件和请求

#### 关键方法：
- `pipe()`: 主入口方法，处理 Open WebUI 请求
- `build_messages_with_tools_and_prompts()`: 构建包含工具信息的消息
- `build_llm_request()`: 构建 LLM API 请求

## 框架使用指南

### 1. 基础配置

1. **创建配置文件**：在 `data/mcp_config.json` 中配置 MCP 服务器
   ```json
   {
     "mcpServers": {
       "server1": {
         "command": "python",
         "args": ["server1.py"],
         "env": {}
       },
       "server2": {
         "command": "python",
         "args": ["server2.py"],
         "env": {}
       }
     }
   }
   ```

2. **配置环境变量**：在 `.env` 文件中设置必要的变量
   ```
   OPENAI_API_KEY=your_key
   OPENAI_API_BASE=http://host.docker.internal:11434/v1
   ```

### 2. 扩展框架

#### 添加新工具服务器

1. 实现一个 MCP 服务器，提供工具和/或提示
2. 在 `mcp_config.json` 中添加服务器配置
3. 框架会自动发现服务器提供的工具和提示

#### 自定义系统提示

修改 `Valves` 类中的 `SYSTEM_PROMPT` 字段：
```python
SYSTEM_PROMPT: str = Field(
    default="""Your custom system prompt here...
    {tools_desc}""",
    description="Custom system prompt"
)
```

#### 添加自定义事件处理

在 `Pipe` 类中添加新的事件处理方法：
```python
async def emit_custom_event(self, data: dict):
    """Emit custom event"""
    await self.__current_event_emitter__(
        {"type": "custom_type", "data": data}
    )
```

### 3. 实现自定义管道

1. 继承 `Pipe` 类并重写方法：
```python
class CustomPipe(Pipe):
    async def pipe(self, body: dict, **kwargs) -> str:
        # Custom processing logic
        return await super().pipe(body, **kwargs)
```

2. 注册管道：
```python
def pipes(self) -> list[dict[str, str]]:
    return [
        {"id": "custom_pipe", "name": "Custom Pipe"},
        *super().pipes()
    ]
```

## 最佳实践

1. **错误处理**：始终在工具调用周围添加错误处理
2. **日志记录**：使用框架提供的 logger 记录关键事件
3. **资源清理**：确保在 `cleanup()` 方法中释放所有资源
4. **性能优化**：对于耗时操作，考虑异步实现

## 示例扩展

### 添加缓存机制

```python
from functools import lru_cache

class CachedPipe(Pipe):
    @lru_cache(maxsize=128)
    async def call_tool(self, tool_name: str, tool_args: Dict) -> str:
        return await super().call_tool(tool_name, tool_args)
```

### 添加认证中间件

```python
class AuthPipe(Pipe):
    async def pipe(self, body: dict, **kwargs) -> str:
        if not self._check_auth():
            await self.emit_status("error", "Unauthorized", True)
            return ""
        return await super().pipe(body, **kwargs)
    
    def _check_auth(self) -> bool:
        # Implement your auth logic
        return True
```

## 调试技巧

1. 启用调试日志：设置 `DEBUG=True` 在 `Valves` 中
2. 检查工具注册：确保工具在 `mcp_config.json` 中正确配置
3. 验证服务器连接：检查日志中的服务器连接状态

通过理解这个框架的结构和扩展点，开发者可以轻松地将其集成到 Open WebUI 中，并根据需要添加自定义功能。