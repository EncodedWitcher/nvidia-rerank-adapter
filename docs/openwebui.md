```
curl -X POST "http://localhost:8080/v1/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-your_api_key_here" \
  -H "X-OpenWebUI-Chat-Id: 12345" \
  -H "X-OpenWebUI-User-Id: user_678" \
  -d '{
    "model": "reranker",
    "query": "如何使用 Open WebUI 进行文档检索？",
    "documents": [
      "文档一：Open WebUI 支持本地模型部署和 RAG 检索。",
      "文档二：使用 embeddings 可以改进检索效果。",
      "文档三：Reranking 可以在检索后重新排序结果以提高相关性。"
    ],
    "top_n": 3
  }'
```
请求体示例：
```
  {
  "model": "reranker",
  "query": "如何使用 Open WebUI 进行文档检索？",
  "documents": [
    "文档一：Open WebUI 支持本地模型部署和 RAG 检索。",
    "文档二：使用 embeddings 可以改进检索效果。",
    "文档三：Reranking 可以在检索后重新排序结果以提高相关性。"
  ],
  "top_n": 3
}
```
请求体中的top_n与documents中元素的数量一定一致

返回值示例：
```
{
  "results": [
    {"index": 2, "relevance_score": 0.90},
    {"index": 0, "relevance_score": 0.75},
    {"index": 1, "relevance_score": 0.60}
  ]
}
```