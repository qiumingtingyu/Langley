# Langley

> AI 原生的个人知识库与学习助手

Langley 是一个面向个人学习与知识管理的 AI Learning Assistant，围绕 **RAG、Agent、长期记忆与可追溯回答**构建。

用户可以导入自己的学习资料，与知识库持续对话，获得带来源引用的回答；Agent 可以按需调用知识检索、网页搜索等工具，并维护跨会话的个人上下文。

## 核心能力

* **知识库与 RAG**：支持文档解析与结构感知 Chunking，使用 BGE-M3 + Qdrant 完成语义检索，并保留标题层级、来源与引用证据。
* **Agent 工具调用**：实现受限的 Model–Tool Loop，可按需调用知识库检索、网页搜索与正文读取等工具。
* **长期上下文记忆**：维护跨会话的 Personal Context，支持用户查看、添加、修改、遗忘以及关闭自动记忆。
* **Eval 与可观测性**：围绕检索质量、回答完整性与 Grounding 建立评测集，并记录 LLM、Tool、Retrieval 等关键执行信息。
* **可靠执行**：回答生成、工具调用与检索采用明确的执行状态和失败恢复机制，持久化事实由 MySQL 管理，Qdrant 作为可重建的向量索引。

## 技术架构

```mermaid
flowchart LR
    UI[Vue 3] --> API[FastAPI]
    API --> AGENT[LangGraph Agent]
    API --> DB[(MySQL)]
    API --> KB[Knowledge Base]

    AGENT --> TOOLS[Knowledge / Web Tools]
    KB --> EMBED[BGE-M3]
    EMBED --> QD[(Qdrant)]
    QD --> AGENT
```

MySQL 保存应用的持久化业务数据，Qdrant 负责语义检索索引，LangGraph 负责 Agent 执行与工具编排。

## 技术栈

* Backend：FastAPI / MySQL
* Agent：LangGraph / Qwen
* Retrieval：BGE-M3 / Qdrant
* Observability：LangSmith / Structured Logging
* Frontend：Vue 3 / Vite / TypeScript

## Quick Start

需要 Python 3.12、[uv](https://docs.astral.sh/uv/)、Node.js 24、npm 和 Docker Desktop。

### 1. 启动 MySQL 与 Qdrant

```powershell
docker compose up -d mysql qdrant
docker compose ps
```

### 2. 配置环境变量

```powershell
$env:LANGLEY_DATABASE_URL = 'mysql+asyncmy://root:langley-local-root-password@127.0.0.1:3306/langley'
$env:LANGLEY_LOCAL_USER_ID = '1'
$env:LANGLEY_QWEN_API_KEY = 'set-only-in-your-shell'
$env:LANGLEY_QWEN_BASE_URL = 'provider-issued-openai-compatible-base-url'
```

更多配置见 [`.env.example`](.env.example)。

### 3. 初始化后端

```powershell
uv sync --locked --group retrieval
uv run alembic upgrade head
uv run python -m langley.bootstrap
```

### 4. 启动服务

```powershell
uv run uvicorn langley.main:create_app --factory --reload
```

另开一个 PowerShell 窗口：

```powershell
Set-Location frontend
npm ci
npm run dev -- --host 127.0.0.1
```

打开 Vite 输出的地址即可使用 Langley。
