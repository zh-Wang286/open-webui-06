# arithmetic_assistant.py
from langchain_openai import AzureChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import tools_condition, ToolNode
import os
import logging

class ArithmeticAssistant:
    def __init__(self):
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing ArithmeticAssistant...")
        
        # ======================
        # API CONFIGURATION
        # ======================
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "7218515241f04d98b3b5d9869a25b91f")
        self.azure_endpoint = os.getenv("AZURE_ENDPOINT", "https://nnitasia-openai-01-ins.openai.azure.com/")
        self.openai_api_version = os.getenv("OPENAI_API_VERSION", "2023-09-01-preview")
        self.model_name_gpt_4o = os.getenv("MODEL_NAME", "NNITAsia-GPT-4o")
        
        self.logger.info("API configuration loaded")
        
        # ======================
        # TOOL DEFINITIONS
        # ======================
        self.tools = [self.add, self.multiply, self.divide]
        self.logger.info(f"Tools registered: {[tool.__name__ for tool in self.tools]}")
        
        # ======================
        # LLM SETUP
        # ======================
        self.llm = AzureChatOpenAI(
            openai_api_key=self.openai_api_key,
            azure_endpoint=self.azure_endpoint,
            deployment_name=self.model_name_gpt_4o,
            openai_api_version=self.openai_api_version,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools, parallel_tool_calls=False)
        self.logger.info("LLM initialized with tools")
        
        # ======================
        # SYSTEM MESSAGE
        # ======================
        self.sys_msg = SystemMessage(
            content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
        )
        self.logger.info("System message configured")
        
        # ======================
        # GRAPH CONSTRUCTION
        # ======================
        self.react_graph = self._build_graph()
        self.logger.info("ReAct graph built and compiled")
        
        self.logger.info("ArithmeticAssistant initialization complete")

    # ======================
    # ARITHMETIC TOOLS
    # ======================
    @staticmethod
    def multiply(a: int, b: int) -> int:
        """Multiply a and b."""
        result = a * b
        logging.info(f"Executing multiply({a}, {b}) = {result}")
        return result

    @staticmethod
    def add(a: int, b: int) -> int:
        """Adds a and b."""
        result = a + b
        logging.info(f"Executing add({a}, {b}) = {result}")
        return result

    @staticmethod
    def divide(a: int, b: int) -> float:
        """Divide a and b."""
        result = a / b
        logging.info(f"Executing divide({a}, {b}) = {result}")
        return result

    # ======================
    # GRAPH CONSTRUCTION
    # ======================
    def _build_graph(self):
        """Build and configure the LangGraph state machine."""
        self.logger.info("Building state graph...")
        
        builder = StateGraph(MessagesState)
        
        # Define nodes
        builder.add_node("assistant", self._assistant_node)
        builder.add_node("tools", ToolNode(self.tools))
        self.logger.info("Nodes added to graph")
        
        # Define edges
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            tools_condition,
        )
        builder.add_edge("tools", "assistant")
        self.logger.info("Edges configured in graph")
        
        compiled_graph = builder.compile()
        self.logger.info("Graph compilation complete")
        return compiled_graph

    def _assistant_node(self, state: MessagesState):
        """Node that handles assistant responses and tool calls."""
        self.logger.info("Processing assistant node...")
        messages = [self.sys_msg] + state["messages"]
        self.logger.debug(f"Input messages to LLM: {messages}")
        
        response = self.llm_with_tools.invoke(messages)
        self.logger.debug(f"LLM response: {response}")
        
        return {"messages": [response]}

    # ======================
    # PUBLIC INTERFACE
    # ======================
    def run(self, query: str):
        """Execute the arithmetic assistant with a user query."""
        self.logger.info(f"Processing query: '{query}'")
        
        messages = [HumanMessage(content=query)]
        result = self.react_graph.invoke({"messages": messages})
        
        final_message = result["messages"][-1].content
        self.logger.info(f"Final result: {final_message}")
        
        return final_message

def interactive_session(assistant):
    """Run an interactive session with the assistant."""
    print("\nArithmetic Assistant - Interactive Mode")
    print("Type 'exit' or 'quit' to end the session\n")
    
    while True:
        try:
            query = input("You: ")
            
            if query.lower() in ['exit', 'quit', "q"]:
                print("Goodbye!")
                break
                
            if not query.strip():
                print("Please enter a valid query")
                continue
                
            result = assistant.run(query)
            print(f"Assistant: {result}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            logging.error(f"Error processing query: {e}")
            print(f"Sorry, an error occurred: {e}")

if __name__ == "__main__":
    try:
        assistant = ArithmeticAssistant()
        
        # Start interactive session
        interactive_session(assistant)
        
    except Exception as e:
        logging.error(f"Failed to initialize assistant: {e}")
        print(f"Failed to initialize assistant: {e}")