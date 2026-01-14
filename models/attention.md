# 注意力机制
## 原模型
### 单头注意力机制 SignalHeadAttnetion
#### 作用
处理不同类型的诊断编码（如共现诊断、邻居诊断和无关诊断）之间的交互关系
#### 输入
- 查询（q）：q = torch.vstack([no_embeddings[m2_index], unrelated_embeddings[m3_index]])
- 键（k）
- 值（v）:v = torch.vstack([co_embeddings[m2_index], co_embeddings[m3_index]]) 
- h_m23 = self.activation(self.single_head_attention(q, q, v))
  - q 是由 no_embeddings[m2_index] 和 unrelated_embeddings[m3_index] 拼接而成，表示新出现的邻居诊断和无关诊断的嵌入表示。
  - v 是由 co_embeddings[m2_index] 和 co_embeddings[m3_index] 拼接而成，表示与新出现的邻居诊断和无关诊断相关的共现诊断嵌入。
  - 通过让 q 和 k 相同，模型可以学习新出现的邻居诊断和无关诊断之间的相互作用。这类似于自注意力机制，帮助模型理解这些新诊断之间的依赖关系。
  - 这种设置意味着模型通过这些新出现的诊断嵌入来关注它们与共现诊断之间的关系。换句话说，q 表示的是当前时间步中新出现的诊断信息，而这些信息将用于关注共现诊断（co_embeddings）。
#### 工作流程
- 对于共现疾病
- 对于邻居和无关疾病
### DotProductAttention

### GRU


# 修改方法
## 1、引入zo_embeddings的多头注意力机制
将 zo_embeddings 作为查询（query）或键值对（key/value）的一部分，利用注意力机制结合当前的诊断嵌入 (co_embeddings)，捕捉检查数据和诊断之间的相互关系。
```python
# 构建查询、键和值
if zo_embeddings is not None:
    zo_query = zo_embeddings  # 检查特征作为查询
    combined_key = torch.cat([co_embeddings, no_embeddings, unrelated_embeddings], dim=0)
    combined_value = torch.cat([co_embeddings, no_embeddings, unrelated_embeddings], dim=0)

    # 使用多头注意力机制学习关系
    zo_output = self.multi_head_attention(zo_query, combined_key, combined_value)
    # 结合诊断的嵌入特征
    output = torch.cat([output, zo_output], dim=-1) if output is not None else zo_output

```
## 2、增加交叉注意力机制
使用zo_embeddings和co_embeddings作为交叉注意力的查询和键值，直接捕捉他们之间的依赖关系
```python
self.cross_attention = CrossAttentionLayer(query_size=zo_embeddings.size(-1), key_size=co_embeddings.size(-1), value_size=co_embeddings.size(-1))

# 在 forward 中使用交叉注意力
if zo_embeddings is not None:
    cross_attention_output = self.cross_attention(zo_embeddings, co_embeddings, co_embeddings)
    output = torch.cat([output, cross_attention_output], dim=-1) if output is not None else cross_attention_output

```
## 3、增加特征选择模块
为模型增加特征选择机制，让模型在 zo_embeddings 中动态选择最重要的特征。
```python
self.zo_feature_selector = nn.Linear(zo_embeddings.size(-1), 1)
# 在 forward 中选择重要的检查特征
if zo_embeddings is not None:
    zo_weights = torch.sigmoid(self.zo_feature_selector(zo_embeddings))  # 权重
    zo_selected = zo_embeddings * zo_weights
    zo_output = torch.sum(zo_selected, dim=1)  # 加权求和
    output = torch.cat([output, zo_output], dim=-1) if output is not None else zo_output
```





