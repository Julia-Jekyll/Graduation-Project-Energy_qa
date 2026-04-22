import os
import sys

# 仅添加一次项目根目录到路径（避免冗余）
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

# 仅导入一次（规范导入）
from utils.vector_db_manager import EnergyVectorDB

def simple_query():
    """极简检索：输入问题 → 查看向量库是否有匹配结果"""
    # 初始化向量库（自动连接本地已有的energy_vector_db）
    vector_db = EnergyVectorDB()
    print("===== 能源知识库检索工具 =====")
    print("提示：输入问题后按回车检索，输入 'exit'/'退出' 退出")
    
    while True:
        try:
            # 获取用户输入
            query = input("\n请输入检索问题：").strip()
            # 支持多格式退出
            if query.lower() in ["exit", "退出"]:
                print("退出检索工具")
                break
            if not query:
                print("⚠️ 问题不能为空，请重新输入")
                continue
            
            # 核心检索逻辑（捕获检索异常）
            results = vector_db.retrieve_knowledge(query)
            
            # 输出结果
            if results:
                print(f"\n✅ 检索到 {len(results)} 条匹配结果：")
                for i, res in enumerate(results, 1):
                    print(f"\n【第{i}条】相似度：{res['score']}")
                    print(f"内容：{res['text']}")
                    print(f"来源：{res['source']}")
            else:
                print("\n❌ 向量库中未检索到匹配结果")
        except Exception as e:
            print(f"\n❌ 检索出错：{str(e)}，请重试")

def simple_query_number():
    """极简检索（带数据量统计）：输入问题 → 查看向量库是否有匹配结果"""
    # 初始化向量库
    vector_db = EnergyVectorDB()
    
    # 新增：打印向量库数据量（处理KeyError）
    try:
        stats = vector_db.get_collection_stats()
        count = stats.get("count", "未知")  # 用get避免KeyError
        print(f"📊 向量库当前数据量：{count} 条")
    except Exception as e:
        print(f"\n❌ 获取数据量失败：{str(e)}")
    
    print("===== 能源知识库检索工具（带数据统计） =====")
    print("提示：输入问题后按回车检索，输入 'exit'/'退出' 退出")
    
    while True:
        try:
            query = input("\n请输入检索问题：").strip()
            if query.lower() in ["exit", "退出"]:
                print("退出检索工具")
                break
            if not query:
                print("⚠️ 问题不能为空，请重新输入")
                continue
            
            results = vector_db.retrieve_knowledge(query)
            if results:
                print(f"\n✅ 检索到 {len(results)} 条匹配结果：")
                for i, res in enumerate(results, 1):
                    print(f"\n【第{i}条】相似度：{res['score']}")
                    print(f"内容：{res['text']}")
                    print(f"来源：{res['source']}")
            else:
                print("\n❌ 向量库中未检索到匹配结果")
        except Exception as e:
            print(f"\n❌ 检索出错：{str(e)}，请重试")

if __name__ == "__main__":
    # 可选择调用simple_query或simple_query_number
    # simple_query()
    simple_query_number()