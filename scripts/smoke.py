"""复原冒烟：含漏读标记与划痕亮点（杂点）的栅格复原。

可直接运行（不依赖服务）；verify 流程在服务健康后通过 BASE_URL 走 HTTP，
同时保留对核心算法的直测。用法::

    python scripts/smoke.py            # 直测求解器
    BASE_URL=http://web:8000 python scripts/smoke.py   # 走 HTTP

共三个场景：旧版无误差带请求回归、误差带内零残差、误差带外超限计残差。
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.solver import reconstruct  # noqa: E402

BOUNDS = {
    "origin": ([-3, 3], [-3, 3]),
    "row_vector": ([-3, 3], [-3, 3]),
    "col_vector": ([-3, 3], [-3, 3]),
}
VB = {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}}


def build_case():
    """4x4 栅格，O=(0,0)，行向量 A=(3,0)，列向量 B=(0,3)，det=9。

    故意漏读 3 个格位，并混入 2 个划痕亮点；再给 3 个标记施加分量 ≤1 的扰动。
    """
    true_cells = {
        (r, c): (3 * r, 3 * c) for r in range(4) for c in range(4)
    }
    missing = {(1, 1), (2, 3), (3, 0), (0, 3)}  # 漏读 4 格
    # 抖动均为曼哈顿 1：任何其他基向量若最大残差同为 1，也必须在众多
    # 精确点上付出更大残差和，真栅格凭第三级目标（残差总和）唯一胜出
    jitter = {(0, 2): (1, 0), (2, 1): (0, -1), (3, 3): (-1, 0)}
    points = []
    pid = 1
    for r in range(4):
        for c in range(4):
            if (r, c) in missing:
                continue
            x, y = true_cells[(r, c)]
            if (r, c) in jitter:
                dx, dy = jitter[(r, c)]
                x += dx
                y += dy
            points.append((pid, x, y))
            pid += 1
    # 两个划痕亮点（杂点），故意打散顺序
    points.extend(
        [
            (90, 17, -5),
            (91, -8, 14),
        ]
    )
    # 打乱坐标顺序，验证“无序坐标恢复”
    import random

    random.Random(42).shuffle(points)
    return points, missing


def expected_payload(points):
    return {
        "points": [{"id": i, "x": x, "y": y} for i, x, y in points],
        "rows": 4,
        "cols": 4,
        "max_outliers": 2,
        "tolerance": 1,
        "origin_bounds": {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
        "row_vector_bounds": {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
        "col_vector_bounds": {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}},
    }


def check_result(result):
    assert result["solvable"] is True, result.get("reason")
    p = result["parameters"]
    assert p["origin"] == [0, 0], p
    assert p["row_vector"] == [3, 0], p
    assert p["col_vector"] == [0, 3], p
    assert p["determinant"] == 9

    obj = result["objective"]
    assert obj["discarded_count"] == 2, obj
    assert obj["max_manhattan_residual"] == 1, obj
    assert obj["total_manhattan_residual"] == 3, obj  # 三个抖动点各 1

    adopted = [a for a in result["assignments"] if a["adopted"]]
    discarded = [a for a in result["assignments"] if not a["adopted"]]
    assert len(adopted) == 12
    assert len(discarded) == 2
    assert {a["id"] for a in discarded} == {90, 91}

    # 每个被采用标记落回真实格位
    true_xy_to_rc = {(3 * r, 3 * c): (r, c) for r in range(4) for c in range(4)}
    cells = set()
    for a in adopted:
        assert max(abs(a["residual"][0]), abs(a["residual"][1])) <= 1
        pr, pc = true_xy_to_rc[tuple(a["predicted"])]
        assert (a["row"], a["col"]) == (pr, pc)
        cells.add((a["row"], a["col"]))
    assert len(cells) == len(adopted), "两个标记占用了同一格位"

    # 弃点证据
    for d in result["discarded"]:
        assert d["id"] in (90, 91)
        assert d["nearest_cell"] is not None
        assert d["nearest_inf_residual"] > 1
        assert d["cells_within_tolerance"] == []
    print(
        f"  弃点 {[d['id'] for d in result['discarded']]}，"
        f"目标 = (k={obj['discarded_count']}, "
        f"max={obj['max_manhattan_residual']}, sum={obj['total_manhattan_residual']})"
    )


# ---------------------------------------------------------------------------
# 场景二：低信噪标记带轴向量测不确定度（误差带）
# ---------------------------------------------------------------------------
def build_band_case():
    """3x3 栅格 O=(0,0)、A=(2,0)、B=(0,2)，9 个标记全部读到。

    两个低信噪标记只给误差带：

    - id=1 真格位 (0,0)，报告中心 (1,0)、x 半宽 1：预测点 (0,0) 落入
      区间 [0,2]，有效残差 0（误差带内零残差）；
    - id=9 真格位 (2,2)=(4,4)，报告中心 (6,4)、x 半宽 1：预测点 (4,4)
      越出区间 [5,7] 一个单位，有效残差 1（误差带外超限计入目标）。
    """
    points = []
    pid = 1
    for r in range(3):
        for c in range(3):
            points.append((pid, 2 * r, 2 * c))
            pid += 1
    points[0] = (1, 1, 0, (1, 0))   # 误差带内
    points[8] = (9, 6, 4, (1, 0))   # 误差带外 1 个单位
    return points


def band_payload(points):
    markers = []
    for t in points:
        i, x, y = t[0], t[1], t[2]
        hx, hy = t[3] if len(t) >= 4 else (0, 0)
        marker = {"id": i, "x": x, "y": y}
        if len(t) >= 4:
            marker["coordinate_uncertainty"] = {"x": hx, "y": hy}
        markers.append(marker)
    return {
        "points": markers,
        "rows": 3,
        "cols": 3,
        "max_outliers": 0,
        "tolerance": 1,
        "origin_bounds": VB,
        "row_vector_bounds": VB,
        "col_vector_bounds": VB,
    }


def check_band_result(result):
    assert result["solvable"] is True, result.get("reason")
    p = result["parameters"]
    assert (p["origin"], p["row_vector"], p["col_vector"]) == (
        [0, 0],
        [2, 0],
        [0, 2],
    ), p
    obj = result["objective"]
    assert obj["discarded_count"] == 0, obj
    assert obj["max_manhattan_residual"] == 1, obj
    assert obj["total_manhattan_residual"] == 1, obj

    by_id = {a["id"]: a for a in result["assignments"]}
    adopted = list(by_id.values())
    cells = {(a["row"], a["col"]) for a in adopted}
    assert len(cells) == 9, "两个标记占用了同一格位"

    # 误差带内：原始偏差非零，但有效残差归零
    a1 = by_id[1]
    assert (a1["row"], a1["col"]) == (0, 0)
    assert a1["predicted"] == [0, 0]
    assert a1["residual"] == [1, 0]
    assert a1["effective_residual"] == [0, 0]
    assert a1["manhattan_residual"] == 0
    assert a1["observation_interval"] == [[0, 2], [0, 0]]
    assert a1["coordinate_uncertainty"] == [1, 0]

    # 误差带外：只计越出区间的 1 个单位
    a9 = by_id[9]
    assert (a9["row"], a9["col"]) == (2, 2)
    assert a9["predicted"] == [4, 4]
    assert a9["residual"] == [2, 0]
    assert a9["effective_residual"] == [1, 0]
    assert a9["manhattan_residual"] == 1

    # 逐点有效残差与汇总目标同一口径
    assert max(a["manhattan_residual"] for a in adopted) == obj[
        "max_manhattan_residual"
    ]
    assert sum(a["manhattan_residual"] for a in adopted) == obj[
        "total_manhattan_residual"
    ]
    print(
        "  id=1 误差带内有效残差 [0,0]；id=9 误差带外有效残差 [1,0]，"
        f"目标 = (k=0, max=1, sum=1)"
    )


def _submit(payload, direct_points, direct_kwargs):
    base_url = os.environ.get("BASE_URL")
    if base_url:
        import urllib.request

        req = urllib.request.Request(
            base_url.rstrip("/") + "/api/wafer-grids/reconstruct",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read()), f"HTTP ({base_url})"
    return reconstruct(direct_points, **direct_kwargs), "直测求解器"


def main():
    print("[场景 1] 旧版请求回归：漏读 4 格 + 2 划痕亮点（无 coordinate_uncertainty）")
    points, _ = build_case()
    result, via = _submit(
        expected_payload(points),
        points,
        dict(rows=4, cols=4, tolerance=1, max_outliers=2, bounds=BOUNDS),
    )
    print(f"==> 通过 {via}")
    check_result(result)
    print("==> 场景 1 通过：旧版请求结果与既有版本一致")

    print("[场景 2] 低信噪误差带：带内零残差 + 带外超限计残差")
    band_points = build_band_case()
    result, via = _submit(
        band_payload(band_points),
        band_points,
        dict(rows=3, cols=3, tolerance=1, max_outliers=0, bounds=BOUNDS),
    )
    print(f"==> 通过 {via}")
    check_band_result(result)
    print("==> 场景 2 通过：误差带口径正确（量测不确定度未被误判为冲突）")

    print("==> 冒烟全部通过")


if __name__ == "__main__":
    main()
