"""
title: LangGraph Calculator Pipe
author: Cascade (Refactored from test02.py)
version: 0.2.0
license: MIT
requirements: pydantic>=2.0.0, langgraph>=0.2.30, langchain-core>=0.1.50
environment_variables: [] # None specifically required for this basic version
Supports: [Addition Calculation Example using LangGraph]
"""

import os
import re
import json
import asyncio
import time
import logging
from typing import Dict, List, Any, Tuple, Optional, Annotated, TypedDict, Literal, Union, AsyncGenerator, Callable

from pydantic import BaseModel, Field

# LangGraph & Langchain Core
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # For potential future stateful operations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- LangGraph State Definition ---

class CalculatorState(TypedDict):
    """State for the LangGraph Calculator Agent."""
    raw_input: str           # User's raw input string
    numbers: List[float]     # Numbers extracted from input
    result: Optional[float]  # Calculation result
    error: Optional[str]     # Any error message during processing
    # We keep tool_usage_count for demonstration, though less critical without confirmation
    tool_usage_count: Dict[str, int]

# --- Pipe Class Implementation ---

class Pipe:
    """
    An OpenWebUI Pipe component implementing a simple calculator using LangGraph.
    This version processes input, parses numbers, calculates the sum, and returns the result.
    The interactive confirmation step from the original script is removed for web compatibility.
    """

    # --- Configuration (Valves) ---
    class Valves(BaseModel):
        """Configuration settings for the Calculator Pipe."""
        DEBUG: bool = Field(default=False, description="Enable debug logging")
        MAX_TOOL_USE: int = Field(default=3, ge=1, description="Maximum usage count per internal tool/node")
        emit_interval: float = Field(default=0.5, ge=0, description="Min interval for status emits (sec)")
        enable_status_indicator: bool = Field(default=True, description="Enable status indicator emits")

    # --- Configuration (Valves) ---

    # --- Initialization ---
    def __init__(self):
        self.type = "pipe"
        self.id = "langgraph_calculator_pipe"
        self.name = "LangGraph Calculator"
        self.valves = self.Valves()
        self.last_emit_time = 0
        # Set logger level based on config
        if self.valves.DEBUG:
            log.setLevel(logging.DEBUG)
            log.debug(f"Pipe '{self.name}' running in DEBUG mode.")
        else:
            log.setLevel(logging.INFO)

        # Build the LangGraph agent
        self.graph = self._build_graph()
        log.info(f"Pipe '{self.name}' (ID: {self.id}) initialized with LangGraph.")
        log.debug(f"Graph Structure:\n{self.graph.get_graph().draw_mermaid()}")


    # --- Required Pipe Methods ---

    def pipes(self) -> List[dict]:
        """Declare the 'models' or capabilities provided by this pipe."""
        return [
            {
                "id": self.id, # Use pipe ID directly as it represents one capability
                "name": self.name,
                # Add other relevant metadata if applicable
                "supports_functions": False,
                "supports_vision": False,
            },
        ]

    async def pipe(self, body: Dict, **kwargs) -> AsyncGenerator[str, None]:
        """Core processing method for the Pipe."""
        emitter = kwargs.get("__event_emitter__")
        user_info = kwargs.get("__user__") # Example access, not used here

        log.debug(f"Pipe '{self.id}' received request. Body keys: {list(body.keys())}, User: {user_info}")
        await self._emit_status(emitter, "info", "Calculator request received...")

        try:
            # 1. Extract input from the last user message
            messages: List[Dict] = body.get("messages", [])
            if not messages or messages[-1].get("role") != "user":
                await self._emit_status(emitter, "error", "No user input found.", True)
                yield "错误: 未提供用户输入"
                return

            raw_input = messages[-1].get("content", "")
            if not isinstance(raw_input, str) or not raw_input.strip():
                 await self._emit_status(emitter, "error", "Empty user input.", True)
                 yield "错误: 提供的用户输入为空"
                 return

            log.info(f"Processing input: '{raw_input}'")
            await self._emit_status(emitter, "info", f"Processing: '{raw_input[:50]}...'")

            # 2. Prepare initial state for LangGraph
            initial_state: CalculatorState = {
                "raw_input": raw_input,
                "numbers": [],
                "result": None,
                "error": None,
                "tool_usage_count": {}
            }

            # 3. Stream LangGraph execution events
            final_state = None
            try:
                async for event in self.graph.astream_events(initial_state, version="v1"):
                    kind = event["event"]
                    # Optional: Use tags for more granular control if needed
                    # tags = event.get("tags", [])
                    name = event.get("name") # Node name

                    log.debug(f"Graph Event: {kind}, Name: {name}, Data: {event['data']}")

                    if kind == "on_chain_start":
                        await self._emit_status(emitter, "info", f"Starting calculation graph...")
                    elif kind == "on_chain_end":
                        if "output" in event["data"]:
                            final_state = event["data"]["output"] # Get the final state from the output
                        await self._emit_status(emitter, "info", "Calculation graph finished.")
                    elif kind == "on_chat_model_start": # Example if using LLM nodes
                        await self._emit_status(emitter, "info", f"Calling model...")
                    elif kind == "on_tool_start": # Adapt if using Langchain tools
                        await self._emit_status(emitter, "info", f"Executing tool: {name}...")
                    elif kind == "on_tool_end":
                        await self._emit_status(emitter, "info", f"Tool finished: {name}.")
                    # Track node execution for status
                    elif kind == "on_node_start":
                        await self._emit_status(emitter, "info", f"Running step: {name}...")
                    elif kind == "on_node_end":
                        await self._emit_status(emitter, "info", f"Step finished: {name}.")
            except Exception as e:
                log.error(f"Error during graph execution: {e}", exc_info=True)
                yield f"错误: 图执行过程中发生错误: {e}"
                return


            # 4. Process final state and yield result
            log.debug(f"Graph final state: {final_state}")
            if final_state:
                try:
                    # 详细输出最终状态的完整内容，便于调试
                    log.info(f"Final state keys: {list(final_state.keys()) if isinstance(final_state, dict) else 'Not a dict'}")
                    log.info(f"Final state full content: {final_state}")
                    
                    # 处理列表类型的final_state
                    # 从日志可以看出，final_state是一个包含节点状态字典的列表
                    # 我们需要提取最后一个节点的状态，通常是output_node
                    actual_state = None
                    if isinstance(final_state, list):
                        # 尝试找到output_node的状态
                        for state_dict in final_state:
                            if isinstance(state_dict, dict) and 'output_node' in state_dict:
                                actual_state = state_dict['output_node']
                                break
                            # 如果没有找到output_node，使用最后一个节点的状态
                            elif isinstance(state_dict, dict) and len(state_dict) > 0:
                                # 获取第一个键对应的值
                                first_key = next(iter(state_dict))
                                actual_state = state_dict[first_key]
                    else:
                        # 如果不是列表，直接使用
                        actual_state = final_state
                    
                    # 确保actual_state是字典
                    if not isinstance(actual_state, dict):
                        log.error(f"Could not extract a valid state dictionary from final_state: {actual_state}")
                        yield "错误: 无法从图执行结果中提取有效状态"
                        return
                    
                    # 使用get方法访问，与_output_node保持一致
                    if actual_state.get("error"):
                        error_message = actual_state["error"]
                        log.error(f"Calculation failed: {error_message}")
                        await self._emit_status(emitter, "error", f"Error: {error_message}", True)
                        # 直接返回错误文本，这是OpenWebUI期望的格式
                        yield f"错误: {error_message}"
                    elif actual_state.get("result") is not None:
                        result = actual_state["result"]
                        numbers = actual_state.get("numbers", [])
                        log.info(f"Processing result: {result}, numbers: {numbers}")
                        
                        # 确保numbers是列表且至少有两个元素
                        if isinstance(numbers, list) and len(numbers) >= 2:
                            output_message = f"{numbers[0]} + {numbers[1]} = {result}"
                        else:
                            output_message = f"计算结果: {result}"
                        log.info(f"Calculation successful: {output_message}")
                        await self._emit_status(emitter, "info", "Calculation complete.", True)
                        # 直接返回文本内容，这是OpenWebUI期望的格式
                        yield output_message
                        return  # 添加return确保在返回结果后结束函数
                    else:
                        # Should ideally have result or error, but handle unexpected state
                        log.warning("Graph finished without a result or explicit error.")
                        await self._emit_status(emitter, "warning", "Calculation finished with unclear outcome.", True)
                        yield "处理完成，但未得到明确结果。"
                except Exception as e:
                    log.error(f"Error processing final state: {e}", exc_info=True)
                    yield f"处理结果时出错: {e}"
            else:
                log.error("Graph stream ended without providing a final state.")
                await self._emit_status(emitter, "error", "Internal error: Graph did not complete correctly.", True)
                yield "内部处理错误: 图执行未返回最终状态"

        except Exception as e:
            log.error(f"Unhandled error in pipe '{self.id}': {e}", exc_info=True)
            await self._emit_status(emitter, "critical", f"An unexpected error occurred: {e}", True)
            # 确保返回的是字符串，这是OpenWebUI期望的格式
            yield f"服务器发生意外错误: {e}"


    # --- LangGraph Node Implementations (as private methods) ---

    def _update_tool_count(self, state: CalculatorState, tool_name: str) -> bool:
        """Updates tool count and checks limit. Returns True if limit exceeded."""
        count = state['tool_usage_count'].get(tool_name, 0) + 1
        state['tool_usage_count'][tool_name] = count
        log.debug(f"Tool '{tool_name}' used {count} times.")
        if count > self.valves.MAX_TOOL_USE:
            log.warning(f"Tool '{tool_name}' usage limit ({self.valves.MAX_TOOL_USE}) exceeded.")
            state['error'] = f"Processing limit exceeded for step '{tool_name}'."
            return True
        return False

    def _input_node(self, state: CalculatorState) -> CalculatorState:
        """Node: Processes the initial raw input."""
        log.info(f"[Node:input] Processing input: '{state['raw_input']}'")
        if 'tool_usage_count' not in state: state['tool_usage_count'] = {}
        if self._update_tool_count(state, "input_node"): return state # Stop if limit exceeded
        # Basic validation could happen here
        if not state['raw_input']:
             state['error'] = "Input cannot be empty."
        return state

    def _parse_node(self, state: CalculatorState) -> CalculatorState:
        """Node: Extracts numbers from the raw input."""
        if state.get("error"): return state # Skip if previous error
        raw_input = state['raw_input']
        log.info(f"[Node:parse] Parsing numbers from: '{raw_input}'")
        if self._update_tool_count(state, "parse_node"): return state

        try:
            # Use regex to find numbers (including decimals and negatives)
            numbers_str = re.findall(r'-?\d+\.?\d*', raw_input)
            numbers = [float(num) for num in numbers_str]

            if len(numbers) < 2:
                log.warning(f"Found fewer than 2 numbers ({numbers}). Cannot perform addition.")
                state['error'] = "Could not find at least two numbers in the input."
                # Fallback or specific error handling could be added here
            else:
                 # Keep only the first two numbers found
                state['numbers'] = numbers[:2]
                log.info(f"Parsed numbers: {state['numbers']}")

        except ValueError as e:
            log.error(f"Error converting parsed strings to float: {e}", exc_info=True)
            state['error'] = f"Invalid number format found in input."
        except Exception as e:
            log.error(f"Unexpected error during parsing: {e}", exc_info=True)
            state['error'] = "An internal error occurred during input parsing."

        return state

    def _calculate_node(self, state: CalculatorState) -> CalculatorState:
        """Node: Performs the addition."""
        if state.get("error") or not state.get('numbers'): return state # Skip if error or no numbers
        numbers = state['numbers']
        log.info(f"[Node:calculate] Calculating sum for: {numbers}")
        if self._update_tool_count(state, "calculate_node"): return state

        try:
            if len(numbers) >= 2:
                state['result'] = sum(numbers) # Calculate sum of the (first two) numbers
                log.info(f"Calculation result: {state['result']}")
            else:
                # This case should ideally be caught by parse_node, but good to be defensive
                 log.warning("Calculate node reached with insufficient numbers.")
                 state['error'] = "Internal error: Calculation step requires at least two numbers."
        except Exception as e:
            log.error(f"Unexpected error during calculation: {e}", exc_info=True)
            state['error'] = "An internal error occurred during calculation."

        return state

    def _output_node(self, state: CalculatorState) -> CalculatorState:
        """Node: Prepares the final output message (or error)."""
        # This node now primarily marks the end and ensures state is ready.
        # The actual yielding happens in the pipe method based on final state.
        log.info("[Node:output] Reached output node.")
        if self._update_tool_count(state, "output_node"): return state

        # 输出完整状态信息便于调试
        log.info(f"Output node state keys: {list(state.keys())}")
        log.info(f"Output node full state: {state}")
        
        if state.get("error"):
            log.info(f"Output node received error state: {state['error']}")
        elif state.get("result") is not None:
            log.info(f"Output node received result: {state['result']}")
            # 确保结果和数字在状态中正确设置
            if "numbers" not in state or not state["numbers"]:
                log.warning("Numbers missing in state, this might cause issues")
        else:
            log.warning("Output node reached with neither result nor error set.")
            # Assign a generic error if state is unexpected
            state['error'] = state.get('error', "Calculation finished with an unknown outcome.")

        return state # Return final state

    # --- Graph Building ---

    def _build_graph(self) -> StateGraph:
        """Builds the LangGraph StateGraph for the calculator."""
        graph = StateGraph(CalculatorState)

        # Add nodes
        graph.add_node("input_node", self._input_node)
        graph.add_node("parse_node", self._parse_node)
        graph.add_node("calculate_node", self._calculate_node)
        graph.add_node("output_node", self._output_node) # Final processing node

        # Define edges (simplified linear flow)
        graph.set_entry_point("input_node")
        graph.add_edge("input_node", "parse_node")
        graph.add_edge("parse_node", "calculate_node")
        graph.add_edge("calculate_node", "output_node")
        graph.add_edge("output_node", END) # End after output node prepares state

        # Compile the graph
        # Could add memory saver here if state needs persistence between pipe calls
        # memory = MemorySaver()
        # return graph.compile(checkpointer=memory)
        return graph.compile()


    # --- Helper Methods ---

    async def _emit_status(self, emitter: Optional[Callable], level: str, message: str, done: bool = False):
        """Helper to send status updates, respecting interval and enabling flag."""
        if not emitter or not self.valves.enable_status_indicator:
            # log.debug(f"Status emission skipped (emitter: {bool(emitter)}, enabled: {self.valves.enable_status_indicator})")
            return

        current_time = time.time()
        # Throttle emissions unless it's the final 'done' message
        if not done and (current_time - self.last_emit_time < self.valves.emit_interval):
            # log.debug(f"Status emission throttled: '{message}'")
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
            log.error(f"Failed to emit status (level='{level}', msg='{message}'): {e}", exc_info=True)


# Example of how to potentially run this outside OpenWebUI for testing (optional)
async def run_pipe_standalone():
    pipe_instance = Pipe()
    # Make Valves debug True for more logs
    # pipe_instance.valves.DEBUG = True
    # log.setLevel(logging.DEBUG)

    # Mock emitter
    async def mock_emitter(event):
        print(f"EVENT: {event}")

    # Mock body
    mock_body = {
        "model": pipe_instance.id,
        "messages": [
            {"role": "user", "content": "Calculate 3.14 plus -1.5"}
        ],
        "stream": True # Test streaming output
    }

    print("\n--- Running Pipe Standalone ---")
    async for chunk in pipe_instance.pipe(mock_body, __event_emitter__=mock_emitter):
        print(f"CHUNK: {chunk}")
    print("--- Pipe Finished ---")
