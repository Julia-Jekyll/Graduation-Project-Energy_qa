# utils/knowledge_processor.py
import os
from pypdf import PdfReader

def clean_invalid_unicode_chars(text):
    """清理无法UTF-8编码的无效Unicode字符"""
    if not text:
        return ""
    return ''.join(
        char for char in text
        if char.isprintable() or char in '\n\r\t'
        and (0x0000 <= ord(char) <= 0x10FFFF)
        and not (0xD800 <= ord(char) <= 0xDFFF)
    )

def read_file_to_text(file_path):
    """仅读取PDF文件为纯文本（清理无效字符）"""
    try:
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"
            return clean_invalid_unicode_chars(text)
        else:
            print(f"不支持的文件格式：{file_ext}")
            return ""
    except Exception as e:
        print(f"读取PDF失败 {file_path}：{str(e)}")
        return ""

def split_text_by_semantic(text, chunk_size=500, chunk_overlap=50):
    """按语义分块（保留原逻辑）"""
    if not text:
        return []
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk + para) > chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = current_chunk[-chunk_overlap:] + para
            else:
                for i in range(0, len(para), chunk_size - chunk_overlap):
                    chunk = para[i:i+chunk_size]
                    chunks.append(chunk)
                current_chunk = ""
        else:
            current_chunk += para + "\n"
    if current_chunk:
        chunks.append(current_chunk)
    return [c.strip() for c in chunks if c.strip()]

def batch_process_knowledge(root_dir=r"D:\桌面\毕设\代码相关\data\energy_knowledge\case",  
        output_dir=r"D:\桌面\毕设\代码相关\data\energy_knowledge\case\processed_knowledge"):
    """批量处理PDF（修复编码写入问题）"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录：{output_dir}")
    
    file_count = 0
    pdf_count = 0
    processed_count = 0
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_count += 1
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            if file_ext != ".pdf":
                continue
            pdf_count += 1
            
            text = read_file_to_text(file_path)
            if not text:
                print(f"跳过空/读取失败的PDF：{file_path}")
                continue
            
            chunks = split_text_by_semantic(text)
            if not chunks:
                print(f"PDF无有效可分块内容：{file_path}")
                continue
            
            # 写入文件时增强编码容错
            file_name = os.path.splitext(file)[0]
            for idx, chunk in enumerate(chunks):
                output_file = os.path.join(output_dir, f"{file_name}_chunk{idx}.txt")
                with open(output_file, "w", encoding="utf-8", errors="ignore") as f:
                    f.write(f"来源PDF：{file_path}\n")
                    f.write(f"块索引：{idx}\n")
                    f.write(f"内容：\n{chunk}\n")
            
            processed_count += 1
            print(f"处理完成：{file_path} → 生成{len(chunks)}个块")
    
    print("\n" + "="*60)
    print(f"扫描文件总数：{file_count}")
    print(f"其中PDF文件数：{pdf_count}")
    print(f"成功处理PDF数：{processed_count}")
    print(f"分块文件保存至：{output_dir}")

if __name__ == "__main__":
    batch_process_knowledge(
        root_dir=r"D:\桌面\毕设\代码相关\data\energy_knowledge\case",  # PDF所在目录
        output_dir=r"D:\桌面\毕设\代码相关\data\energy_knowledge\case\processed_knowledge"  # 输出目录
    )