#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
加法计算LangGraph流程示例
包含输入、解析、确认、计算和输出节点
"""

import re
import json
from typing import Dict, List, Any, Tuple, Optional, Annotated, TypedDict, Literal, Union

# LangGraph 相关导入
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

# 控制台日志记录
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义状态类型
class CalculatorState(TypedDict):
    """加法计算器的状态类型"""
    raw_input: str  # 用户原始输入
    numbers: List[float]  # 提取的数字
    confirmed: bool  # 用户是否确认计算
    result: Optional[float]  # 计算结果
    tool_usage_count: Dict[str, int]  # 工具使用次数统计

# 节点函数定义
def input_node(state: CalculatorState) -> CalculatorState:
    """输入节点：接收用户原始输入"""
    logger.info(f"输入节点接收到用户输入: {state['raw_input']}")
    
    # 初始化工具使用计数（如果不存在）
    if 'tool_usage_count' not in state:
        state['tool_usage_count'] = {}
    
    # 更新工具使用计数
    tool_name = "input_node"
    state['tool_usage_count'][tool_name] = state['tool_usage_count'].get(tool_name, 0) + 1
    logger.info(f"工具 {tool_name} 已使用 {state['tool_usage_count'][tool_name]} 次")
    
    # 检查工具使用次数是否超限
    if state['tool_usage_count'][tool_name] > 3:
        logger.warning(f"工具 {tool_name} 使用次数超过限制，流程将结束")
        state['result'] = None
        state['confirmed'] = False
        return state
    
    return state

def parse_node(state: CalculatorState) -> CalculatorState:
    """解析节点：提取输入中的数字"""
    raw_input = state['raw_input']
    logger.info(f"解析节点处理输入: {raw_input}")
    
    # 更新工具使用计数
    tool_name = "parse_node"
    state['tool_usage_count'][tool_name] = state['tool_usage_count'].get(tool_name, 0) + 1
    logger.info(f"工具 {tool_name} 已使用 {state['tool_usage_count'][tool_name]} 次")
    
    # 检查工具使用次数是否超限
    if state['tool_usage_count'][tool_name] > 3:
        logger.warning(f"工具 {tool_name} 使用次数超过限制，流程将结束")
        state['result'] = None
        state['confirmed'] = False
        return state
    
    # 使用正则表达式提取数字
    numbers = [float(num) for num in re.findall(r'-?\d+\.?\d*', raw_input)]
    
    # 如果提取到的数字少于2个，设置为默认值
    if len(numbers) < 2:
        logger.warning(f"未能提取到足够的数字，当前提取到: {numbers}")
        # 确保至少有两个数字用于加法运算
        while len(numbers) < 2:
            numbers.append(0.0)
    
    # 只取前两个数字
    numbers = numbers[:2]
    logger.info(f"解析结果: {numbers}")
    
    state['numbers'] = numbers
    return state

def confirm_node(state: CalculatorState) -> CalculatorState:
    """确认节点：要求用户确认是否执行计算"""
    numbers = state['numbers']
    logger.info(f"确认节点展示数字: {numbers}")
    
    # 更新工具使用计数
    tool_name = "confirm_node"
    state['tool_usage_count'][tool_name] = state['tool_usage_count'].get(tool_name, 0) + 1
    logger.info(f"工具 {tool_name} 已使用 {state['tool_usage_count'][tool_name]} 次")
    
    # 检查工具使用次数是否超限
    if state['tool_usage_count'][tool_name] > 3:
        logger.warning(f"工具 {tool_name} 使用次数超过限制，流程将结束")
        state['result'] = None
        state['confirmed'] = False
        return state
    
    print("\n" + "=" * 50)
    print(f"确认计算: {numbers[0]} + {numbers[1]}")
    print("是否执行计算? (y/n)")
    
    # 获取用户输入
    user_input = input().strip().lower()
    confirmed = user_input == "y"
    
    logger.info(f"用户确认结果: {confirmed}")
    state['confirmed'] = confirmed
    
    return state

def calculate_node(state: CalculatorState) -> CalculatorState:
    """计算节点：执行加法运算"""
    if not state['confirmed']:
        logger.info("用户拒绝执行计算")
        state['result'] = None
        return state
    
    numbers = state['numbers']
    logger.info(f"计算节点处理数字: {numbers}")
    
    # 更新工具使用计数
    tool_name = "calculate_node"
    state['tool_usage_count'][tool_name] = state['tool_usage_count'].get(tool_name, 0) + 1
    logger.info(f"工具 {tool_name} 已使用 {state['tool_usage_count'][tool_name]} 次")
    
    # 检查工具使用次数是否超限
    if state['tool_usage_count'][tool_name] > 3:
        logger.warning(f"工具 {tool_name} 使用次数超过限制，流程将结束")
        state['result'] = None
        return state
    
    # 执行加法运算
    result = numbers[0] + numbers[1]
    logger.info(f"计算结果: {result}")
    
    state['result'] = result
    return state

def output_node(state: CalculatorState) -> CalculatorState:
    """输出节点：返回最终结果"""
    logger.info("输出节点处理结果")
    
    # 更新工具使用计数
    tool_name = "output_node"
    state['tool_usage_count'][tool_name] = state['tool_usage_count'].get(tool_name, 0) + 1
    logger.info(f"工具 {tool_name} 已使用 {state['tool_usage_count'][tool_name]} 次")
    
    # 检查工具使用次数是否超限
    if state['tool_usage_count'][tool_name] > 3:
        logger.warning(f"工具 {tool_name} 使用次数超过限制，流程将结束")
        return state
    
    # 根据计算结果生成输出
    if state.get('result') is not None:
        print("\n" + "=" * 50)
        print(f"计算结果: {state['numbers'][0]} + {state['numbers'][1]} = {state['result']}")
    else:
        if not state.get('confirmed', True):
            print("\n" + "=" * 50)
            print("计算已取消: 用户拒绝执行计算")
        else:
            print("\n" + "=" * 50)
            print("计算失败: 未能获取有效结果")
    
    return state

# 条件路由函数
def route_based_on_confirmation(state: CalculatorState) -> Literal["calculate_node", "output_node"]:
    """根据用户确认结果决定下一步"""
    if state.get('confirmed', False):
        logger.info("用户确认执行计算，路由到计算节点")
        return "calculate_node"
    else:
        logger.info("用户拒绝执行计算，路由到输出节点")
        return "output_node"

def check_tool_usage_limit(state: CalculatorState) -> Literal["end", "continue"]:
    """检查工具使用次数是否超限"""
    # 检查是否有任何工具使用次数超过3次
    for tool, count in state.get('tool_usage_count', {}).items():
        if count > 3:
            logger.warning(f"工具 {tool} 使用次数超过限制，流程将结束")
            return "end"
    return "continue"

# 构建图
def build_graph() -> StateGraph:
    """构建加法计算流程图"""
    # 创建图
    graph = StateGraph(CalculatorState)
    
    # 设置初始状态
    graph.set_entry_point("input_node")
    
    # 添加节点
    graph.add_node("input_node", input_node)
    graph.add_node("parse_node", parse_node)
    graph.add_node("confirm_node", confirm_node)
    graph.add_node("calculate_node", calculate_node)
    graph.add_node("output_node", output_node)
    
    # 连接节点
    graph.add_edge("input_node", "parse_node")
    graph.add_edge("parse_node", "confirm_node")
    
    # 条件分支：根据用户确认结果决定下一步
    graph.add_conditional_edges(
        "confirm_node",
        route_based_on_confirmation,
        {
            "calculate_node": "calculate_node",
            "output_node": "output_node"
        }
    )
    
    graph.add_edge("calculate_node", "output_node")
    graph.add_edge("output_node", END)
    
    # 编译图
    return graph.compile()

# 主函数
def main():
    """主函数"""
    # 构建图
    graph = build_graph()
    
    # 打印图结构
    print("\n" + "=" * 50)
    print("加法计算流程图结构:")
    print(graph.get_graph().draw_mermaid())
    
    # 获取用户输入
    print("\n" + "=" * 50)
    print("请输入两个数字(例如: '计算 3 加 5'):")
    user_input = input().strip()
    
    # 初始状态
    initial_state = {
        "raw_input": user_input,
        "numbers": [],
        "confirmed": False,
        "result": None,
        "tool_usage_count": {}
    }
    
    # 运行图
    print("\n" + "=" * 50)
    print("开始执行加法计算流程...")
    
    # 收集所有状态更新
    final_state = None
    for state in graph.stream(initial_state):
        # 每个状态更新时打印当前节点
        if state.get("current_node"):
            print(f"当前节点: {state['current_node']}")
        # 保存最后一个完整状态
        if all(key in state for key in ["raw_input", "numbers", "confirmed"]):
            final_state = state
    
    # 打印最终状态
    print("\n" + "=" * 50)
    print("执行完成!")
    print("最终状态:")
    if final_state:
        print(f"- 原始输入: {final_state.get('raw_input', '未知')}")
        print(f"- 提取数字: {final_state.get('numbers', [])}")
        print(f"- 用户确认: {final_state.get('confirmed', False)}")
        print(f"- 计算结果: {final_state.get('result', '未计算')}")
        print(f"- 工具使用统计: {json.dumps(final_state.get('tool_usage_count', {}), ensure_ascii=False, indent=2)}")
    else:
        print("无法获取完整状态")

if __name__ == "__main__":
    main()