# utils/vector_db_manager.py
import os
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import numpy as np

# 配置项
EMBEDDING_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
VECTOR_DB_PATH = "./data/energy_vector_db"       # 向量库存储路径
COLLECTION_NAME = "energy_knowledge"             # 向量库集合名
CHUNK_SIZE = 300                                 # 适配能源政策文本的分块大小
CHUNK_OVERLAP = 50
TOP_K = 5                                        # 检索返回Top5结果
SCORE_THRESHOLD = 0.35                           # 相似度过滤阈值

class EnergyVectorDB:
    """能源领域向量数据库管理器：构建/查询/更新向量库"""
    def __init__(self):
        self.client = chromadb.Client(
            Settings(
                persist_directory=VECTOR_DB_PATH,
                anonymized_telemetry=False,        
                is_persistent=True
            )
        )
        
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu"  
        )
        
        # 创建/获取集合，指定余弦相似度（适配BGE模型）
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={
                "description": "能源领域知识库向量库（政策/技术/案例）",
                "hnsw:space": "cosine"  
            }
        )

    def text_to_embedding(self, text: str) -> List[float]:
        """单段文本转为向量（适配新版SentenceTransformer）"""
        if not text or len(text.strip()) < 5: 
            return []
        enhanced_text = f"为这个句子生成表示以用于检索相关文章：{text.strip()}"
        embedding = self.embedding_model.encode(
            enhanced_text,
            normalize_embeddings=True  
        )
        return embedding.tolist()

    def add_knowledge_to_db(self, knowledge_chunks: List[Dict]) -> int:
        """批量添加知识块到向量库"""
        if not knowledge_chunks:
            print("无有效知识块，跳过添加")
            return 0
        
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        
        for chunk in knowledge_chunks:
            chunk_id = chunk.get("id")
            chunk_text = chunk.get("text")
            chunk_meta = chunk.get("metadata", {})
            
            if not chunk_id or not chunk_text:
                continue
            
            embedding = self.text_to_embedding(chunk_text)
            if not embedding:
                continue
            
            ids.append(chunk_id)
            texts.append(chunk_text)
            metadatas.append(chunk_meta)
            embeddings.append(embedding)
        
        # 批量入库
        added_count = len(ids)
        if added_count > 0:
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f" 成功添加 {added_count} 个知识块到向量库")
            return added_count
        else:
            print(" 无有效数据可添加")
            return 0

    def retrieve_knowledge(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """高精度检索：带相似度阈值过滤"""
        query_embedding = self.text_to_embedding(query)
        if not query_embedding:
            return []
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,
            include=["documents", "metadatas", "distances"]
        )
        
        retrieved_results = []
        for idx in range(len(results["documents"][0])):
            doc = results["documents"][0][idx]
            meta = results["metadatas"][0][idx] if results["metadatas"] else {}
            score = 1 - results["distances"][0][idx]
            
            if score < SCORE_THRESHOLD:
                continue
            
            retrieved_results.append({
                "text": doc,
                "score": round(score, 4),
                "source": meta.get("source", "未知来源"),  # 读取metadata中的PDF来源
                "category": meta.get("category", "未知分类")
            })
        
        # 按相似度降序，最终返回top_k条
        retrieved_results = sorted(retrieved_results, key=lambda x: x["score"], reverse=True)[:top_k]
        return retrieved_results

    def clear_db(self):
        """清空向量库（重新入库前用）"""
        self.client.delete_collection(COLLECTION_NAME)
        self.collection = self.client.create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "能源领域知识库向量库", "hnsw:space": "cosine"}
        )
        print("向量库已清空")

    def get_collection_stats(self) -> Dict:
        """获取向量库统计信息"""
        try:
            count = self.collection.count()
            return {
                "count": count,
                "path": VECTOR_DB_PATH,
                "name": COLLECTION_NAME
            }
        except Exception as e:
            print(f"❌ 获取统计信息失败：{e}")
            return {"count": 0, "path": VECTOR_DB_PATH, "name": COLLECTION_NAME}

    def persist_db(self):
        """兼容旧代码的空实现（新版无需手动persist）"""
        pass