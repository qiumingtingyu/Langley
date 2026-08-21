# 工具调用
工具调用需要保留原始 Markdown 字节。

## 示例
1. **准备** `name`
2. 记录参数

> 引用说明。

```python
def run_tool(name, args):
    # not a heading
    return registry[name](**args)
```

| 字段 | 说明 |
| --- | --- |
| name | 工具名称 |

正文保持 **bold** 与 `inline code`。
