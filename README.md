# Active Olfaction Paper

本仓库用于维护论文《面向间歇羽流的双触须主动嗅觉采样与强化学习搜索方法》的 Markdown 稿件。

## 文件结构

- `chapters/00_front_matter.md`：标题、摘要、符号与缩略语
- `chapters/01_introduction.md`：第1章 引言
- `chapters/02_hardware_system.md`：第2章 自适应嗅觉触须硬件系统设计
- `chapters/03_rl_algorithm.md`：第3章 基于深度强化学习的自主嗅觉搜索算法
- `chapters/04_simulation_optimization.md`：第4章 仿真环境与参数优化
- `chapters/05_experiments_discussion.md`：第5章 实验设计与讨论
- `chapters/06_conclusion.md`：第6章 结论与展望
- `chapters/07_appendices.md`：附录
- `chapters/08_references.md`：参考文献
- `chapters/09_todo.md`：后续补充清单
- `paper_full.md`：由各章节按顺序合并得到的完整 Markdown 版本
- `scripts/build_paper_full.py`：重建完整稿，章节文件是正文的唯一来源

## 分支策略

`main` 保存可随时阅读和合并的完整论文版本。每个章节使用独立分支修改：

- `chapter/00-front-matter`
- `chapter/01-introduction`
- `chapter/02-hardware`
- `chapter/03-algorithm`
- `chapter/04-simulation`
- `chapter/05-experiments`
- `chapter/06-conclusion`
- `chapter/07-appendices-references`

在对应分支中原则上只修改该章节文件。完成一章后创建 Pull Request 合并到 `main`。这样不同章节可以并行修改，并尽量降低合并冲突。

## 推荐工作流

```bash
git switch chapter/03-algorithm
# 编辑 chapters/03_rl_algorithm.md
git add chapters/03_rl_algorithm.md
git commit -m "revise chapter 3 algorithm"
git push
```

章节完成后在 GitHub 上创建 PR：

```text
chapter/03-algorithm -> main
```

合并后，如其他章节分支后续还要继续工作，建议同步 `main`：

```bash
git switch chapter/04-simulation
git merge main
```

## 写作约定

1. 不在没有实验数据的情况下补写虚构结果；保留 `【待补】` 标记。
2. 数学公式使用 GitHub 可渲染的 LaTeX：行内公式写作 `$...$`，独立公式写作 `$$...$$`；不要用裸下划线或转义下划线表示上下标。
3. 术语首次出现时给出中英文与缩写，后文保持一致。
4. 对算法、奖励函数和仿真参数的修改，应同步核对 `dual-whisker-rl` 项目代码。
5. 最终投稿前再统一处理图表编号、交叉引用、参考文献格式和 Word/LaTeX 排版。

修改章节后运行以下命令同步完整稿：

```bash
python scripts/build_paper_full.py
```
