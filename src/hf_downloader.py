import os
import time
import shutil
from datetime import datetime
from typing import Optional, Tuple, Dict
from huggingface_hub import HfApi, snapshot_download
import huggingface_hub
from tqdm import tqdm

print("huggingface_hub version:", huggingface_hub.__version__)
print("huggingface_hub path:", huggingface_hub.__file__)

api = HfApi()
# 缓存项目类型和信息的字典
repo_info_cache: Dict[str, Tuple[str, object, str]] = {}

def format_repo_info(repo_info, repo_type, api_endpoint):
    # 构造完整API地址
    if repo_type == 'model':
        api_url = f"{api_endpoint.rstrip('/')}/api/models/{repo_info.id}"
    elif repo_type == 'dataset':
        api_url = f"{api_endpoint.rstrip('/')}/api/datasets/{repo_info.id}"
    elif repo_type == 'space':
        api_url = f"{api_endpoint.rstrip('/')}/api/spaces/{repo_info.id}"
    else:
        api_url = api_endpoint
    # 获取SHA、许可证、总体积
    sha = getattr(repo_info, 'sha', '-')
    license_ = getattr(repo_info, 'license', '-')
    # 尝试获取项目总体积（部分类型有 size 字段）
    size = getattr(repo_info, 'size', None)
    if size is not None:
        size_str = format_size(size)
    else:
        size_str = '-'
    return (
        f"\n==== 项目详情 ===="
        f"\nAPI地址: {api_url}"
        f"\n项目名称: {repo_info.id}"
        f"\n项目类型: {repo_type}"
        f"\nSHA值: {sha}"
        f"\n许可证: {license_}"
        f"\n项目总体积: {size_str}"
        f"\n作者: {getattr(repo_info, 'author', '-') }"
        f"\n创建时间: {getattr(repo_info, 'created_at', '-') }"
        f"\n最后更新: {getattr(repo_info, 'last_modified', '-') }"
        f"\n下载量: {getattr(repo_info, 'downloads', '-') }"
        f"\n点赞数: {getattr(repo_info, 'likes', '-') }"
        f"\n私有: {'是' if getattr(repo_info, 'private', False) else '否'}"
    )

def detect_repo_type(repo_id: str) -> Tuple[Optional[str], Optional[object], Optional[str]]:
    """
    检测项目类型并返回项目信息
    
    参数:
        repo_id: 项目ID (如 username/repo)
        
    返回:
        元组 (项目类型, 项目信息, API端点) 或 (None, None, None) 如果失败
    """
    # 检查缓存
    if repo_id in repo_info_cache:
        return repo_info_cache[repo_id]
        
    # 尝试检测模型或数据集类型
    for repo_type in ['model', 'dataset']:
        try:
            repo_info = api.repo_info(repo_id, repo_type=repo_type)
            result = (repo_type, repo_info, api.endpoint)
            repo_info_cache[repo_id] = result  # 缓存结果
            return result
        except Exception as e:
            print(f'❌ 尝试 {repo_type} 类型失败，错误类型: {type(e).__name__}, 详细信息: {str(e)}')
            
    # 尝试检测space类型
    try:
        api_official = HfApi(endpoint="https://huggingface.co")
        repo_info = api_official.space_info(repo_id)
        result = ('space', repo_info, api_official.endpoint)
        repo_info_cache[repo_id] = result  # 缓存结果
        return result
    except Exception as e:
        print(f'❌ 尝试 space 类型失败，错误类型: {type(e).__name__}, 详细信息: {str(e)}')
        
    return None, None, None

def get_dir_size_and_count(path):
    total_size = 0
    total_count = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            if os.path.isfile(fp):
                total_size += os.path.getsize(fp)
                total_count += 1
    return total_size, total_count

def get_subdir_size_and_count(path, subfolder=None):
    # 统计指定子目录或文件的体积和数量
    if not subfolder:
        return get_dir_size_and_count(path)
    abs_path = os.path.join(path, subfolder)
    if os.path.isfile(abs_path):
        return os.path.getsize(abs_path), 1
    elif os.path.isdir(abs_path):
        return get_dir_size_and_count(abs_path)
    else:
        return 0, 0

def format_size(size):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"

def get_file_list(repo_id, repo_type="model", revision="main", subfolder=None, endpoint=None):
    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    try:
        files = api.list_repo_files(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision
        )
        if subfolder:
            # 只保留指定子目录下的文件
            files = [f for f in files if f.startswith(subfolder)]
        return files
    except Exception as e:
        print("获取文件列表失败，错误信息：", e)
        return []

def main():
    print('==== Hugging Face 下载器 ====' )
    while True:
        repo_id = input('\n请输入 Hugging Face 项目名称（如 username/model，直接回车退出）：').strip()
        if not repo_id:
            print('\n再见！')
            break
        print('\n[信息] 正在查询项目信息...')
        repo_type, repo_info, api_endpoint = detect_repo_type(repo_id)
        if not repo_type:
            print('\n❌ 未找到该项目，请检查名称是否正确。')
            continue
        print(format_repo_info(repo_info, repo_type, api_endpoint))
        # 以项目名（去掉用户名）命名的文件夹
        project_name = repo_id.split('/')[-1]
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), project_name)
        os.makedirs(save_dir, exist_ok=True)
        print(f'📁 下载路径: {save_dir}')
        revision = input('请输入分支、tag 或 commit（直接回车默认 main）: ').strip() or 'main'
        subfolder = input('请输入要下载的子目录或文件（直接回车下载全部内容）: ').strip()
        # 预下载统计
        while True:
            print('\n[信息] 正在获取将要下载的文件数量...')
            try:
                files = get_file_list(repo_id, repo_type, revision, subfolder)
                print(f'⭐ 重要: 将要下载的文件数量: {len(files)}')
            except Exception as e:
                print(f'❌ 无法预览将要下载的文件列表，下载后会显示实际体积和数量。\n   错误信息: {e}')
            confirm = input('是否下载？（直接回车确认，输入 n 返回主菜单，输入路径则重新查询该路径）: ').strip()
            if confirm == '' or confirm.lower() == 'y':
                break
            elif confirm.lower() == 'n':
                print('\n[提示] 已取消，返回主菜单。')
                break
            else:
                subfolder = confirm
        if confirm.lower() == 'n':
            continue
        print('\n🚩 开始下载...')
        start_time = datetime.now()
        start_ts = time.time()
        try:
            # 确保子目录路径格式正确
            if subfolder and not subfolder.endswith('/'):
                subfolder = subfolder + '/'
                
            kwargs = {
                'repo_id': repo_id,
                'repo_type': repo_type,
                'local_dir': save_dir,
                'local_dir_use_symlinks': False,
                'revision': revision
            }
            if subfolder:
                kwargs['allow_patterns'] = [subfolder + '*']
                
            # 再次验证文件列表
            files = get_file_list(repo_id, repo_type, revision, subfolder)
            if not files:
                print(f'❌ 没有找到匹配的文件: {subfolder}')
                continue
                
            # 下载文件到指定目录
            path = snapshot_download(**kwargs)
            
            # 移动下载的文件到根目录并清理缓存
            cache_dir = os.path.join(save_dir, f"models--{repo_id.replace('/', '--')}")
            if os.path.exists(cache_dir):
                try:
                    # 移动下载的文件到根目录
                    snapshots_dir = os.path.join(cache_dir, "snapshots")
                    if os.path.exists(snapshots_dir):
                        for item in os.listdir(snapshots_dir):
                            item_path = os.path.join(snapshots_dir, item)
                            if os.path.isdir(item_path):
                                for file in os.listdir(item_path):
                                    src = os.path.join(item_path, file)
                                    dst = os.path.join(save_dir, file)
                                    if not os.path.exists(dst):
                                        shutil.move(src, dst)
                    
                    # 清理缓存目录
                    shutil.rmtree(cache_dir)
                    
                    # 将必要的内部目录设置为隐藏
                    if os.path.exists(os.path.join(save_dir, ".git")):
                        try:
                            import ctypes
                            ctypes.windll.kernel32.SetFileAttributesW(os.path.join(save_dir, ".git"), 2)
                        except:
                            pass
                except Exception as e:
                    print(f'⚠️ 无法优化目录结构: {e}')
            end_time = datetime.now()
            end_ts = time.time()
            total_size, total_count = get_dir_size_and_count(path)
            print(f'\n✅ 下载完成！')
            print(f'⭐ 重要: 保存路径: {path}')
            print(f'⭐ 重要: 下载文件数量: {total_count}')
            print(f'⭐ 重要: 下载文件总大小: {format_size(total_size)}')
            print(f'   下载开始时间: {start_time.strftime("%Y-%m-%d %H:%M:%S")}' +
                  f'\n   下载完成时间: {end_time.strftime("%Y-%m-%d %H:%M:%S")}' +
                  f'\n   下载用时: {end_ts - start_ts:.2f} 秒')
        except Exception as e:
            print(f'\n❌ 下载失败，错误信息: {e}')
    input("按回车键关闭窗口...")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n[致命错误] 程序运行中断，错误信息：{e}")
    input("按回车键关闭窗口...")