# Wafer Grid Reconstruction Service

从无序、带唯一编号的整数标记坐标中恢复晶圆栅格：联合选择**原点 O、行基向量 A、
列基向量 B**以及标记到互异格位的分配，容忍漏读（缺标记）与最多两个杂点
（划痕亮点）。低信噪标记可逐点提供 **x / y 轴向量测不确定度（误差带）**，
直接在区间上复原栅格，避免先取中心值而误弃合法标记。

## 量测不确定度（误差带）

`points[*].coordinate_uncertainty` 选填：

```json
{"id": 1, "x": 1, "y": 0, "coordinate_uncertainty": {"x": 1, "y": 0}}
```

`x`/`y` 为 0~3 的非负整数轴向半宽，观测区间是
`[x-hx, x+hx] × [y-hy, y+hy]`；省略该字段时两轴半宽均为 0（与旧版完全
一致）。有效轴向残差为：

```
eff_x = max(|obs_x − pred_x| − hx, 0)
eff_y = max(|obs_y − pred_y| − hy, 0)
```

预测坐标落入观测区间的方向有效残差为 0，仅越出区间的轴向距离计入
`tolerance` 可达判定（`eff ≤ tolerance`）、**最大曼哈顿残差**与**残差总和**。
原始有符号偏差（标记中心 − 预测坐标）仍同时返回，互异格位、正行列式及
字典序裁决保持不变。

## 优化目标（字典序）

对每组候选参数与分配依次最小化：

1. **弃点数**（≤ `max_outliers`）；
2. **最大曼哈顿残差**（按有效残差口径）；
3. **曼哈顿残差总和**（按有效残差口径）；
4. **完整参数与按编号排列的分配序列**（枚举序下的首个最优，保证确定性）。

硬约束：

- `det(A, B) > 0`；
- 采用点预测坐标满足 `eff_x ≤ tolerance 且 eff_y ≤ tolerance`；
- 任意两个标记不得占用同一格位（二分图匹配保证）；
- O / A / B 各分量取自调用方给定的、跨度 ≤ 6 的闭区间。

算法：枚举至多 `7^6` 组整数参数（det 过滤，包围盒扩张半宽 + 邻域计数
预过滤），Kuhn 求最大匹配与瓶颈残差，最小费用流求有效残差和，再逐点贪心 +
后缀可行性检查得到字典序最小分配。

## 运行

```bash
# 端口可配置（默认 8000）
API_PORT=9000 ./verify
```

`verify` 会构建镜像、启动 `web` 服务（带容器健康检查），待服务健康后由一次性
`verify` 容器执行：

1. `pytest` 代码测试（含误差带内零残差、带外超限、非法字段定位、真实冲突
   与不确定度区分）；
2. 复原冒烟（场景 1：4×4 栅格漏读 4 格 + 2 划痕亮点的旧请求回归；
   场景 2：低信噪误差带——带内零残差 + 带外超限计残差；均经 HTTP 提交）；

并以自身退出码汇报（成功 0）。单独启动服务：`API_PORT=9000 docker compose up web`。

## API

`GET /health` → `{"status":"healthy"}`

`POST /api/wafer-grids/reconstruct`：

```json
{
  "points": [{"id": 1, "x": 0, "y": 0}],
  "rows": 4,
  "cols": 4,
  "max_outliers": 2,
  "tolerance": 1,
  "origin_bounds":      {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
  "row_vector_bounds":  {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
  "col_vector_bounds":  {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}}
}
```

约束：7–14 个唯一编号点；行/列 3–7；`max_outliers` 0–2；各区间跨度 ≤ 6；
`coordinate_uncertainty.x/y` 为 0~3 的非负整数（选填）。

成功返回（HTTP 200，`solvable: true`）：`parameters`（原点、两基向量、行列式）、
`objective`（弃点数 / 最大有效残差 / 有效残差和）、`assignments`（逐点格位、
预测坐标、原始偏差 `residual`、有效残差 `effective_residual`、观测区间
`observation_interval` 与半宽回显、曼哈顿有效残差）、`discarded`（弃点证据：
最近格位、按有效口径的最近残差、容差内候选、弃点原因）。

几何上无解时返回 HTTP 200、`solvable: false` 及明确的中文 `reason`
（建议放宽容差/区间或提高弃点上限），可与量测不确定度区分；请求本身不合法
（编号重复、点数越界、区间跨度超 6、半宽越界等）返回 HTTP 422，错误 `loc`
精确到具体点下标与轴，如 `body → points → 3 → coordinate_uncertainty → x`。

## 本地开发

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/smoke.py              # 直测求解器
BASE_URL=http://127.0.0.1:8000 python scripts/smoke.py   # 走 HTTP
```
