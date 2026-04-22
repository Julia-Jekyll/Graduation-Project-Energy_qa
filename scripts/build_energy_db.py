import os
import sys
import re
import uuid
import math
from typing import List, Dict

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)
from utils.vector_db_manager import EnergyVectorDB

# 配置项
KNOWLEDGE_TXT_DIR = "./data/energy_knowledge" 
BATCH_SIZE = 50  # 批量入库大小
CHUNK_SIZE = 300  # 分块大小
CHUNK_OVERLAP = 50  # 分块重叠长度

def extract_source_from_txt(file_path: str) -> str:
    """从txt提取来源，无论成功/失败都仅返回文件名"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            first_lines = f.readlines()[:5]
            for line in first_lines:
                match = re.search(r'来源PDF:\s*(.+)', line.strip())
                if match:
                    pdf_full_path = match.group(1).strip()
                    pdf_filename = os.path.basename(pdf_full_path)
                    pdf_filename = os.path.splitext(pdf_filename)[0]
                    return pdf_filename
        txt_filename = os.path.basename(file_path)
        txt_base_name = re.sub(r'_chunk\d+\.txt$', '', txt_filename)
        return txt_base_name
    except Exception as e:
        print(f" 提取来源失败 {file_path}：{e}")
        txt_filename = os.path.basename(file_path)
        txt_base_name = re.sub(r'_chunk\d+\.txt$', '', txt_filename)
        return txt_base_name

def split_text_by_size(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """纯Python原生文本分块"""
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        if end > text_length:
            end = text_length
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - chunk_overlap
        if start >= text_length:
            break
    return chunks

def split_document_to_chunks(file_path: str) -> List[Dict]:
    """读取txt→净化来源→分块→补充metadata"""
    source_pdf = extract_source_from_txt(file_path)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    content_lines = []
    chunk_index = 0  
    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("块索引:"):
            chunk_index = line_strip.replace("块索引:", "").strip()
            continue
        if line_strip.startswith(("来源PDF:", "内容:")):
            continue
        content_lines.append(line)
    
    content = "".join(content_lines).strip()
    if not content:
        return []
    
    chunks = split_text_by_size(content, CHUNK_SIZE, CHUNK_OVERLAP)
    chunk_list = []
    txt_filename = os.path.basename(file_path)
    txt_base_name = re.sub(r'_chunk\d+\.txt$', '', txt_filename)
    for idx, chunk_text in enumerate(chunks):
        if not chunk_text.strip():
            continue
        chunk_id = f"{txt_base_name}_{idx}_{uuid.uuid4().hex[:8]}"
        chunk_list.append({
            "id": chunk_id,
            "text": chunk_text,
            "metadata": {
                "source": source_pdf,      
                "txt_file": txt_base_name,  
                "chunk_index": chunk_index if chunk_index else idx
            }
        })
    return chunk_list

def format_retrieval_result(result: Dict, similarity: float) -> str:
    """统一输出格式"""
    source = result["metadata"]["source"]
    chunk_index = result["metadata"]["chunk_index"]
    content = result["text"].strip().replace("\n", " ").replace("  ", " ")
    formatted_str = f"相似度：{similarity:.4f}  来源：{source}，块索引：{chunk_index}，内容：{content}"
    return formatted_str

def load_all_chunks_from_dir(txt_dir: str) -> List[Dict]:
    """加载目录下所有知识块（无数量限制）"""
    all_chunks = []
    abs_txt_dir = os.path.join(project_root, txt_dir)
    for root, _, files in os.walk(abs_txt_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                print(f"📄 处理文件：{file_path}")
                chunks = split_document_to_chunks(file_path)
                all_chunks.extend(chunks)
    print(f" 共加载 {len(all_chunks)} 个知识块")
    return all_chunks

def batch_add_to_db(vector_db, all_chunks: List[Dict], batch_size=50):
    """批量入库（带进度提示）"""
    total = len(all_chunks)
    if total == 0:
        print(" 无知识块可入库")
        return
    
    vector_db.clear_db()
    print(f" 开始分批入库，共 {total} 条数据，分 {math.ceil(total/batch_size)} 批处理")
    for i in range(0, total, batch_size):
        batch_chunks = all_chunks[i:i+batch_size]
        batch_num = i//batch_size + 1
        print(f"\n 处理第 {batch_num}/{math.ceil(total/batch_size)} 批，共 {len(batch_chunks)} 条")
        
        add_count = vector_db.add_knowledge_to_db(batch_chunks)
        print(f" 第{batch_num}批成功添加 {add_count} 条")

def main():
    """主流程：加载全量分块→批量入库"""
    vector_db = EnergyVectorDB()
    all_chunks = load_all_chunks_from_dir(KNOWLEDGE_TXT_DIR)
    batch_add_to_db(vector_db, all_chunks, batch_size=BATCH_SIZE)
    stats = vector_db.get_collection_stats()
    print(f"\n 入库完成！向量库当前数据量：{stats['count']} 条，存储路径：{stats['path']}")

if __name__ == "__main__":
    main()