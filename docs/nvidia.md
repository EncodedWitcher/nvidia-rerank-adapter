```
curl --request POST \
     --url https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking \
     --header 'accept: application/json' \
     --header 'authorization: Bearer nvapi-your_api_key_here' \
     --header 'content-type: application/json' \
     --data '
{
  "query": {
    "text": "112345"
  },
  "passages": [
    {
      "text": "114514"
    },
    {
      "text": "1919810"
    },
    {
      "text": "2333333"
    }
  ],
  "truncate": "NONE",
  "model": "nvidia/rerank-qa-mistral-4b"
}
'
```
返回值示例：
```
{
  "rankings": [
    {
      "index": 0,
      "logit": -1.78515625
    }
  ]
}
```