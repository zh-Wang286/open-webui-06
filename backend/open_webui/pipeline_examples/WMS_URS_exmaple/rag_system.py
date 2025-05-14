#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
WMS 需求文档 RAG 系统

该模块实现了基于 WMS 系统用户需求说明文档的 RAG (检索增强生成) 系统。
主要功能包括：
1. 加载和处理 WMS 需求文档
2. 构建向量数据库
3. 提供检索接口，支持基于用户需求的相关文档片段检索

作者: AI 助手
日期: 2025-04-24
"""

import json
import os
import logging
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings
from pydantic import BaseModel, Field

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class WMSRequirementChunk(BaseModel):
    """WMS 需求文档片段"""
    category: str = Field(description="需求类别，如功能性需求、控制需求等")
    item_id: str = Field(description="需求项 ID，如 URS-6.2-4")
    item_name: Optional[str] = Field(description="需求项名称，如上架管理")
    criticality: str = Field(description="重要性，如关键、非关键")
    content: str = Field(description="详细内容")
    
    def to_string(self) -> str:
        """将需求片段转换为字符串格式"""
        item_name_str = f"【{self.item_name}】" if self.item_name else ""
        return f"{self.category} {self.item_id} {item_name_str} ({self.criticality})\n{self.content}"


class WMSDocumentProcessor:
    """WMS 文档处理器，负责加载和处理 WMS 需求文档"""
    
    def __init__(self, file_path: str):
        """
        初始化 WMS 文档处理器
        
        参数:
            file_path: WMS 需求文档路径
        """
        self.file_path = file_path
        self.doc_data = None
        logger.info(f"初始化 WMS 文档处理器，文档路径: {file_path}")
    
    def load_document(self) -> Dict[str, Any]:
        """加载 WMS 需求文档"""
        logger.info(f"加载 WMS 需求文档: {self.file_path}")
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.doc_data = json.load(f)
            logger.info("WMS 需求文档加载成功")
            return self.doc_data
        except Exception as e:
            logger.error(f"加载 WMS 需求文档失败: {e}")
            raise
    
    def extract_requirements(self) -> List[WMSRequirementChunk]:
        """提取 WMS 需求文档中的需求项"""
        if not self.doc_data:
            self.load_document()
        
        requirements = []
        for req in self.doc_data.get("requirements", []):
            try:
                requirement = WMSRequirementChunk(
                    category=req.get("category", ""),
                    item_id=req.get("item_id", ""),
                    item_name=req.get("item_name", ""),
                    criticality=req.get("criticality", ""),
                    content=req.get("content", "")
                )
                requirements.append(requirement)
            except Exception as e:
                logger.error(f"处理需求项失败: {e}, 需求项: {req}")
        
        logger.info(f"成功提取 {len(requirements)} 个需求项")
        return requirements
    
    def get_document_metadata(self) -> Dict[str, Any]:
        """获取文档元数据"""
        if not self.doc_data:
            self.load_document()
        
        metadata = {
            "title": self.doc_data.get("doc_metadata", {}).get("title", ""),
            "version": self.doc_data.get("doc_metadata", {}).get("version", ""),
            "toc": self.doc_data.get("toc", []),
            "purpose": self.doc_data.get("purpose", ""),
            "scope": self.doc_data.get("scope", "")
        }
        
        return metadata


class WMSRAGSystem:
    """WMS RAG 系统，提供基于 WMS 需求文档的检索增强生成功能"""
    
    def __init__(
        self, 
        doc_path: str,
        azure_api_key: Optional[str] = None,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        azure_api_version: Optional[str] = "2023-09-01-preview",
        persist_directory: Optional[str] = "./chroma_db"
    ):
        """
        初始化 WMS RAG 系统
        
        参数:
            doc_path: WMS 需求文档路径
            azure_api_key: Azure OpenAI API 密钥
            azure_endpoint: Azure OpenAI 端点
            azure_deployment: Azure OpenAI 部署名称
            azure_api_version: Azure OpenAI API 版本
            persist_directory: 向量数据库持久化目录
        """
        self.doc_path = doc_path
        self.persist_directory = persist_directory
        
        # 初始化 Azure OpenAI 嵌入模型
        self.azure_api_key = azure_api_key or os.getenv("OPENAI_API_KEY", "7218515241f04d98b3b5d9869a25b91f")
        self.azure_endpoint = azure_endpoint or os.getenv("AZURE_ENDPOINT", "https://nnitasia-openai-01-ins.openai.azure.com/")
        self.azure_deployment = azure_deployment or os.getenv("EMBEDDING_DEPLOYMENT", "NNIT-Ada-3-large")
        self.azure_api_version = azure_api_version or os.getenv("OPENAI_API_VERSION", "2023-09-01-preview")
        
        self.embedding = AzureOpenAIEmbeddings(
            openai_api_key=self.azure_api_key,
            azure_endpoint=self.azure_endpoint,
            deployment=self.azure_deployment,
            openai_api_version=self.azure_api_version,
        )
        
        self.doc_processor = WMSDocumentProcessor(doc_path)
        self.vectorstore = None
        self.doc_metadata = None
        
        logger.info("WMS RAG 系统初始化完成")
    
    def build_vectorstore(self, force_rebuild: bool = False) -> None:
        """
        构建向量数据库
        
        参数:
            force_rebuild: 是否强制重建向量数据库
        """
        # 检查是否已存在向量数据库
        if os.path.exists(self.persist_directory) and not force_rebuild:
            logger.info(f"加载现有向量数据库: {self.persist_directory}")
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding
            )
            return
        
        # 提取需求项
        requirements = self.doc_processor.extract_requirements()
        self.doc_metadata = self.doc_processor.get_document_metadata()
        
        # 转换为 Document 对象
        documents = []
        for req in requirements:
            doc = Document(
                page_content=req.to_string(),
                metadata={
                    "category": req.category,
                    "item_id": req.item_id,
                    "item_name": req.item_name or "",
                    "criticality": req.criticality
                }
            )
            documents.append(doc)
        
        # 创建向量数据库
        logger.info(f"创建向量数据库，文档数量: {len(documents)}")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embedding,
            persist_directory=self.persist_directory
        )
        self.vectorstore.persist()
        logger.info(f"向量数据库创建成功并持久化到: {self.persist_directory}")
    
    def query(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        查询相关需求片段
        
        参数:
            query_text: 查询文本
            top_k: 返回结果数量
            
        返回:
            相关需求片段列表
        """
        if not self.vectorstore:
            logger.warning("向量数据库未初始化，正在构建...")
            self.build_vectorstore()
        
        logger.info(f"执行查询: '{query_text}'")
        results = self.vectorstore.similarity_search_with_relevance_scores(
            query_text, 
            k=top_k
        )
        
        formatted_results = []
        for doc, score in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": score
            })
        
        logger.info(f"查询完成，找到 {len(formatted_results)} 个相关结果")
        return formatted_results
    
    def query_by_requirement_points(self, requirement_points: List[str], top_k: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """
        根据需求点列表查询相关需求片段
        
        参数:
            requirement_points: 需求点列表
            top_k: 每个需求点返回的结果数量
            
        返回:
            需求点与相关需求片段的映射
        """
        results = {}
        for point in requirement_points:
            results[point] = self.query(point, top_k=top_k)
        
        return results
    
    def format_rag_results_for_llm(self, user_input: str, results: List[Dict[str, Any]]) -> str:
        """
        格式化 RAG 结果，用于 LLM 输入
        
        参数:
            user_input: 用户输入
            results: RAG 查询结果
            
        返回:
            格式化后的文本
        """
        formatted_text = f"用户输入的需求：{user_input}\n\n{'='*20}\n\n知识库召回片段：\n\n"
        
        for i, result in enumerate(results, 1):
            formatted_text += f"片段 {i}:\n{result['content']}\n\n"
        
        return formatted_text
    
    def get_document_structure(self) -> Dict[str, Any]:
        """获取文档结构信息"""
        if not self.doc_metadata:
            self.doc_metadata = self.doc_processor.get_document_metadata()
        
        return self.doc_metadata


if __name__ == "__main__":
    # 示例用法
    doc_path = "/data01/open-webui-06/pipeline_examples/WMS-URS-exmaple.py/CS-002-URS-01-WMS系统用户需求说明-20210723.txt"
    rag_system = WMSRAGSystem(doc_path)
    
    # 构建向量数据库
    rag_system.build_vectorstore(force_rebuild=True)
    
    # 测试查询
    test_query = "上架管理和物料位置追踪"
    results = rag_system.query(test_query)
    
    print(f"查询: '{test_query}'")
    print("结果:")
    for i, result in enumerate(results, 1):
        print(f"结果 {i}:")
        print(f"内容: {result['content']}")
        print(f"相关度: {result['relevance_score']}")
        print("-" * 50)
