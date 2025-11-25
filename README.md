# Huggingface 模型下载工具

一个简单高效的Huggingface模型下载工具，支持断点续传、多线程下载等功能。

## 功能特性

✅ 基本模型/数据集/space下载功能  
✅ 断点续传  
✅ 多线程下载  
✅ 下载进度显示  
✅ 模型验证  
✅ 项目信息查询  
✅ 子目录下载  
✅ URL直接下载  
✅ 下载统计信息  

## 安装

### 通过pip安装
```bash
pip install -e .
```

### 安装依赖
```bash
pip install huggingface-hub tqdm
```

## 使用说明

1. 安装完成后，在命令行运行：
```bash
hf-downloader
```
2. 按照提示输入Hugging Face项目名称
3. 可选择下载特定分支、tag或commit
4. 可选择下载整个项目或特定子目录

## 示例

1. 下载整个模型：
```
请输入 Hugging Face 项目名称（如 username/model，直接回车退出）：bert-base-uncased
```

2. 下载特定子目录：
```
请输入 Hugging Face 项目名称（如 username/model，直接回车退出）：bert-base-uncased
请输入要下载的子目录或文件（直接回车下载全部内容）: config.json
```

3. 通过URL直接下载：
```
请输入 Hugging Face 项目名称或URL（直接回车退出）：https://huggingface.co/bert-base-uncased/blob/main/config.json
```

4. 下载特定分支的子目录：
```
请输入 Hugging Face 项目名称（如 username/model，直接回车退出）：bert-base-uncased
请输入分支、tag 或 commit（直接回车默认 main）: v1.0.0
请输入要下载的子目录或文件（直接回车下载全部内容）: config.json
```

## 贡献指南

欢迎提交Pull Request或Issue报告问题。

## 许可证

本项目采用MIT许可证，详情见LICENSE文件。

