# NVIDIA Rerank Adapter

一个基于 FastAPI 的适配器，将 OpenAI 兼容的 rerank API 请求转换为 NVIDIA 的 rerank API 格式。这使您可以在期望 OpenAI 兼容 API 的客户端（如 OpenWebUI）中使用 NVIDIA 强大的重排序模型。

## 功能特性

- 🔄 **格式转换**：自动在 OpenAI/OpenWebUI 和 NVIDIA API 格式之间进行转换
- 🔑 **多密钥支持**：配置多个 NVIDIA API 密钥，自动轮询切换
- 🔁 **自动故障转移**：失败时自动切换到下一个密钥，支持冷却管理
- 📊 **可扩展模型**：轻松添加新的 NVIDIA rerank 模型
- 🐳 **Docker 就绪**：包含 Dockerfile 和 Docker Compose，便于部署
- 📝 **OpenAPI 文档**：内置 Swagger UI 文档

## 快速开始

### 1. 安装

```bash
# 克隆仓库
git clone https://github.com/EncodedWitcher/nvidia-rerank-adapter.git
cd nvidia-rerank-adapter

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

编辑 `.env` 文件，配置您的 NVIDIA API 密钥：

```bash
# 编辑 .env 文件
NVIDIA_API_KEYS=nvapi-your-key-1,nvapi-your-key-2
```

### 3. 运行服务器

```bash
python main.py
```

服务器将在 `http://localhost:8000` 启动。

## 配置选项

| 环境变量 | 描述 | 默认值 |
|---------|------|-------|
| `NVIDIA_API_KEYS` | NVIDIA API 密钥（多个密钥用逗号分隔） | （必需） |
| `DEFAULT_MODEL` | 默认使用的 NVIDIA 模型 | `nvidia/rerank-qa-mistral-4b` |
| `HOST` | 服务器主机地址 | `0.0.0.0` |
| `PORT` | 服务器端口 | `8000` |
| `REQUEST_TIMEOUT` | API 请求超时时间（秒） | `30` |
| `API_KEY` | 客户端认证 API 密钥（可选） | （无） |

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

### GET /v1/models

列出可用的 rerank 模型。

### GET /health

健康检查端点。

### GET /v1/keys/status

获取 API 密钥池状态（如果设置了 API_KEY 则需要认证）。

## 可用模型

| 模型别名 | NVIDIA 模型 |
|---------|------------|
| `reranker` | `nvidia/rerank-qa-mistral-4b` |
| `nvidia/rerank-qa-mistral-4b` | `nvidia/rerank-qa-mistral-4b` |
| `nvidia/nv-rerank-qa-mistral-4b` | `nvidia/nv-rerank-qa-mistral-4b` |
| `nvidia/llama-3.2-nv-rerankqa-1b-v2` | `nvidia/llama-3.2-nv-rerankqa-1b-v2` |

要添加更多模型，请编辑 `config.py` 并在 `MODEL_CONFIGS` 中添加条目。

## Docker 部署

### 方式一：Docker Compose（推荐）

这是最简单的部署方式：

```bash
# 1. 编辑 .env 文件，配置您的 NVIDIA API 密钥
# NVIDIA_API_KEYS=nvapi-your-key-1,nvapi-your-key-2

# 2. 启动服务
docker-compose up -d

# 3. 查看日志
docker-compose logs -f

# 4. 停止服务
docker-compose down
```

### 方式二：直接使用 Docker

```bash
# 构建镜像
docker build -t nvidia-rerank-adapter .

# 运行容器
docker run -d \
  --name nvidia-rerank-adapter \
  -p 8000:8000 \
  -e NVIDIA_API_KEYS=nvapi-your-key-1,nvapi-your-key-2 \
  nvidia-rerank-adapter
```

### Docker Compose 配置示例

```yaml
version: '3.8'

services:
  nvidia-rerank-adapter:
    build: .
    container_name: nvidia-rerank-adapter
    ports:
      - "8000:8000"
    environment:
      - NVIDIA_API_KEYS=${NVIDIA_API_KEYS}
      - DEFAULT_MODEL=${DEFAULT_MODEL:-nvidia/rerank-qa-mistral-4b}
      - HOST=0.0.0.0
      - PORT=8000
      - REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-30}
      - API_KEY=${API_KEY:-}
    env_file:
      - .env
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/health')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

networks:
  rerank-network:
    driver: bridge
```

## API 密钥管理

适配器实现了智能的 API 密钥管理：

- **轮询切换**：密钥按顺序轮换以实现负载分配
- **失败追踪**：跟踪每个密钥的失败次数
- **自动冷却**：连续 3 次失败后，密钥将被冷却 1 分钟
- **自动恢复**：冷却期结束后密钥自动恢复

## 开发指南

### 项目结构

```
nvidia-rerank-adapter/
├── main.py              # FastAPI 应用入口
├── config.py            # 配置和模型映射
├── models.py            # Pydantic 数据模型
├── services/
│   ├── __init__.py
│   ├── key_manager.py   # 多密钥轮询管理器
│   └── nvidia_client.py # NVIDIA API 客户端
├── routers/
│   ├── __init__.py
│   └── rerank.py        # Rerank API 路由
├── .env                 # 环境配置
├── requirements.txt     # Python 依赖
├── Dockerfile           # Docker 构建文件
├── docker-compose.yml   # Docker Compose 配置
└── README.md            # 本文件
```

### 开发模式运行

```bash
# 安装依赖
pip install -r requirements.txt

# 使用热重载运行
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### API 文档

服务器运行后，访问：
- Swagger UI：http://localhost:8000/docs
- ReDoc：http://localhost:8000/redoc

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 致谢

- [NVIDIA AI API](https://build.nvidia.com/) 提供重排序模型
- [FastAPI](https://fastapi.tiangolo.com/) 优秀的 Web 框架
