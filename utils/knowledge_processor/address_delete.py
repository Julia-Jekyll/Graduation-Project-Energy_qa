import os
import re

def replace_source_pdf_line(match):
    """
    替换回调函数：输入正则匹配对象，返回替换后的字符串。
    """
    full_path = match.group(1).strip()   
    filename = os.path.basename(full_path)
    if filename.lower().endswith('.pdf'):
        filename = filename[:-4]        
    return f"来源：{filename}"

def process_file(filepath, encoding='utf-8'):
    """
    处理单个文件：读取、替换、写回。
    返回是否修改了文件。
    """
    try:
        with open(filepath, 'r', encoding=encoding) as f:
            content = f.read()
    except UnicodeDecodeError:
        # 如果 utf-8 解码失败，尝试 gbk
        try:
            with open(filepath, 'r', encoding='gbk') as f:
                content = f.read()
        except Exception as e:
            print(f"无法读取文件 {filepath}，编码问题: {e}")
            return False

    pattern = r'来源PDF：([^\n]+)'
    new_content, count = re.subn(pattern, replace_source_pdf_line, content)

    if count > 0:
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            print(f"已修改文件：{filepath}，替换了 {count} 处。")
            return True
        except Exception as e:
            print(f"写入文件 {filepath} 失败：{e}")
            return False
    else:
        print(f"文件无变化：{filepath}")
        return False

def main():
    # 固定文件夹路径
    folder_path = r"D:\桌面\毕设\代码相关\data\energy_knowledge\technology\processed_knowledge"
    
    if not os.path.isdir(folder_path):
        print(f"错误：文件夹 '{folder_path}' 不存在。")
        return

    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            process_file(filepath)

if __name__ == "__main__":
    main()
    print("所有文件处理完成。")