# Wafer Grid Reconstruction Service

从无序、带唯一编号的整数标记坐标中恢复晶圆栅格：联合选择**原点 O、行基向量 A、
列基向量 B**以及标记到互异格位的分配，容忍漏读（缺标记）与最多两个杂点
（划痕亮点）。

## 优化目标（字典序）

对每组候选参数与分配依次最小化：

1. **弃点数**（≤ `max_outliers`）；
2. **最大曼哈顿残差**；
3. **曼哈顿残差总和**；
4. **完整参数与按编号排列的分配序列**（枚举序下的首个最优，保证确定性）。

硬约束：

- `det(A, B) > 0`；
- 采用点预测坐标逐分量满足 `|dx| ≤ tolerance 且 |dy| ≤ tolerance`；
- 任意两个标记不得占用同一格位（二分图匹配保证）；
- O / A / B 各分量取自调用方给定的、跨度 ≤ 6 的闭区间。

### 轴向不确定度（误差带）

低信噪区域每个标记可在 `coordinate_uncertainty` 中分别给出 x、y 非负整数
半宽 `hx, hy`（取值 0–3；省略时两轴半宽均为 0），观测区间为
`[x-hx, x+hx] × [y-hy, y+hy]`。预测坐标 `(cx, cy)` 的**有效（区间化）残差**为

```
ex = max(0, cx-(x+hx), (x-hx)-cx)
ey = max(0, cy-(y+hy), (y-hy)-cy)
```

即落入观测区间的轴向分量为 0，仅越出区间的轴向距离计入：

- 采用边存在性（容差硬约束）：`ex ≤ tolerance 且 ey ≤ tolerance`；
- 最大曼哈顿残差与残差总和均按 `ex + ey` 计算。

原始 `residual`（预测相对观测中心）继续保留；另返回 `effective_residual`
（扣除误差带后的非负轴向残差）与每点的 `coordinate_uncertainty`。全部标记
省略不确定度时，口径与旧版本完全一致。

算法：枚举至多 `7^6` 组整数参数（包围盒 + 邻域集合两级预过滤，det 过滤），
Kuhn 求最大匹配与瓶颈残差，最小费用流求残差和，再逐点贪心 + 后缀可行性检查
得到字典序最小分配。

## 运行

```bash
# 端口可配置（默认 8000）
API_PORT=9000 ./verify
```

`verify` 会构建镜像、启动 `web` 服务（带容器健康检查），待服务健康后由一次性
`verify` 容器执行：

1. `pytest` 代码测试；
2. 复原冒烟（经 HTTP 提交）：
   - 旧口径回归：4×4 栅格漏读 4 格 + 2 个划痕亮点 + 坐标抖动；
   - 轴向不确定度：误差带内零残差采用、误差带外越限量计入容差与弃点；

并以自身退出码汇报（成功 0）。单独启动服务：`API_PORT=9000 docker compose up web`。

## API

`GET /health` → `{"status":"healthy"}`

`POST /api/wafer-grids/reconstruct`：

```json
{
  "points": [
    {"id": 1, "x": 0, "y": 0},
    {"id": 2, "x": 1, "y": 2, "coordinate_uncertainty": {"x": 1, "y": 0}}
  ],
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
`coordinate_uncertainty.x/y` 为 0–3 的非负整数，选填，非法时 422 错误定位到
具体点（`points[i]`）与具体轴（`coordinate_uncertainty.x/y`）。

成功返回（HTTP 200，`solvable: true`）：`parameters`（原点、两基向量、行列式）、
`objective`（弃点数 / 最大残差 / 残差和）、`assignments`（逐点格位、预测坐标、
残差、有效残差、不确定半宽）、`discarded`（弃点证据：最近格位、最近有效残差、
容差内候选、弃点原因）。`objective` 中的最大/总和曼哈顿残差与
`assignments[*].manhattan_residual` 一律采用有效残差口径。

几何上无解时返回 HTTP 200、`solvable: false` 及明确的中文 `reason`
（建议放宽容差/区间或提高弃点上限）；请求本身不合法（编号重复、点数越界、
区间跨度超 6 等）返回 HTTP 422 并附字段级错误。

## 本地开发

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest -q
python scripts/smoke.py              # 直测求解器
BASE_URL=http://127.0.0.1:8000 python scripts/smoke.py   # 走 HTTP
```
