# NVIDIA Rerank & Embeddings Adapter

一个基于 FastAPI 的适配器，将 OpenAI 兼容的 Rerank 和 Embeddings API 请求转发到 NVIDIA API。支持多密钥轮询、429 自动重试，适用于 OpenWebUI 等客户端。

## 功能特性

- 🔄 **Rerank 格式转换**：自动在 OpenAI/OpenWebUI 和 NVIDIA API 格式之间进行转换
- 📤 **Embeddings 透传**：直接转发 embeddings 请求到 NVIDIA API
-  **多密钥轮询**：配置多个 NVIDIA API 密钥，自动轮询切换
- 🔁 **429 自动重试**：遇到速率限制时自动切换密钥重试，不向下游返回错误
- 🐳 **Docker 就绪**：包含 Dockerfile 和 Docker Compose

## 快速开始

### 1. 配置

复制环境变量模板并配置：

```bash
cp .env.example .env
# 编辑 .env，填入您的 NVIDIA API 密钥
```

### 2. 启动服务

**使用 Docker Compose（推荐）：**

```bash
docker-compose up -d
```

**或直接运行：**

```bash
pip install -r requirements.txt
python main.py
```

服务将在 `http://localhost:8000` 启动。

## 配置选项

| 环境变量 | 描述 | 默认值 |
|---------|------|-------|
| `NVIDIA_API_KEYS` | NVIDIA API 密钥（多个用逗号分隔） | （必需） |
| `DEFAULT_MODEL` | 默认 Rerank 模型 | `nvidia/rerank-qa-mistral-4b` |
| `DEFAULT_EMBEDDING_MODEL` | 默认 Embedding 模型 | `baai/bge-m3` |
| `NVIDIA_EMBEDDINGS_URL` | NVIDIA Embeddings API 地址 | `https://integrate.api.nvidia.com/v1/embeddings` |
| `HOST` | 服务器地址 | `0.0.0.0` |
| `PORT` | 服务器端口 | `8000` |
| `REQUEST_TIMEOUT` | 请求超时（秒） | `30` |
| `RETRY_WAIT_SECONDS` | 所有密钥 429 后的等待时间（秒） | `5` |
| `API_KEY` | 客户端认证密钥（可选） | （无） |

## API 端点

### POST /v1/rerank

根据与查询的相关性对文档进行重新排序。

**请求：**
```json
{
  "model": "reranker",
  "query": "如何使用文档检索？",
  "documents": [
    "文档一：OpenWebUI 支持本地模型部署和 RAG 检索。",
    "文档二：使用 embeddings 可以改进检索效果。",
    "文档三：Reranking 可以在检索后重新排序结果以提高相关性。"
  ],
  "top_n": 3
}
```

**响应：**
```json
{
  "results": [
    {"index": 2, "relevance_score": 0.90},
    {"index": 0, "relevance_score": 0.75},
    {"index": 1, "relevance_score": 0.60}
  ]
}
```

### POST /v1/embeddings

透传 embeddings 请求到 NVIDIA API。

**请求：**
```json
{
  "model": "baai/bge-m3",
  "input": ["Hello world", "How are you?"],
  "encoding_format": "float"
}
```

**响应：** 直接返回 NVIDIA API 的响应。

> 如果请求的模型不存在，会自动回退到 `DEFAULT_EMBEDDING_MODEL`。

### GET /v1/models

列出可用的 rerank 模型。

### GET /health

健康检查端点。

### GET /v1/keys/status

获取 API 密钥池状态（需要认证）。

## API 密钥管理

适配器实现了简单高效的密钥管理：

- **轮询负载均衡**：密钥按顺序轮换使用
- **429 自动重试**：遇到速率限制时切换到下一个密钥
- **等待重试**：所有密钥都 429 时，等待 `RETRY_WAIT_SECONDS` 秒后继续
- **保证成功**：持续重试直到成功，不向下游返回 429 错误

## 项目结构

```
nvidia-rerank-adapter/
├── main.py              # FastAPI 应用入口
├── config.py            # 配置和模型映射
├── models.py            # Pydantic 数据模型
├── services/
│   ├── key_manager.py   # 多密钥轮询管理器
│   └── nvidia_client.py # NVIDIA API 客户端
├── routers/
│   ├── rerank.py        # Rerank API 路由
│   └── embeddings.py    # Embeddings API 路由
├── .env.example         # 环境变量模板
├── docker-compose.yml   # Docker Compose 配置
├── Dockerfile           # Docker 构建文件
└── requirements.txt     # Python 依赖
```

## API 文档

服务器运行后，访问：
- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件。
