# 语见 - 可选手语翻译模型

## 当前部署

| 模型 | 检查点 | 数据集 | 特点 |
|------|--------|--------|------|
| Uni-Sign RGB+Pose | `csl_daily_rgb_pose_slt.pth` | CSL Daily | 当前使用中 |

## 更好的模型选项

根据 Uni-Sign 原论文，以下模型可能效果更好：

### 1. CSL-News 数据集模型 (推荐尝试)
- **特点**: 更大的数据集，更多样化的手语表达
- **下载**: https://huggingface.co/datasets/ZechengLi19/CSL-News
- **优势**: 词汇量更大，泛化能力更强

### 2. How2Sign 数据集模型
- **特点**: 美式手语 (ASL) 数据集
- **适用**: 国际标准手语
- **下载**: https://huggingface.co/ZechengLi19/Uni-Sign

### 3. OpenASL 数据集模型
- **特点**: 大规模 ASL 数据集
- **优势**: 更好的连续手语理解
- **下载**: https://huggingface.co/ZechengLi19/Uni-Sign

### 4. Stage-2 预训练模型
- **特点**: 在大规模数据上预训练后的模型
- **优势**: 比 Stage-3 微调模型更通用
- **注意**: 需要自行训练

## 建议

如果需要更好的准确度：

1. **下载 CSL-News 检查点** (如果作者提供)
2. **训练自己的模型** - 用更多数据微调
3. **考虑其他 SOTA 模型** - 如 Sign2GPT、SignLLM 等
