"""复原冒烟：含漏读标记与划痕亮点（杂点）的栅格复原。

可直接运行（不依赖服务）；verify 流程在服务健康后通过 BASE_URL 走 HTTP，
同时保留对核心算法的直测。用法::

    python scripts/smoke.py            # 直测求解器
    BASE_URL=http://web:8000 python scripts/smoke.py   # 走 HTTP

包含两组场景：

1. 旧口径回归：4x4 栅格漏读 + 划痕亮点 + 抖动（不带轴向不确定度）；
2. 轴向不确定度：误差带内零残差采用、误差带外越限量计入容差/弃点。
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
# 轴向不确定度（误差带）冒烟
# ---------------------------------------------------------------------------
VB3 = {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}}


def _band_base_payload(points, tolerance, max_outliers):
    markers = []
    for t in points:
        m = {"id": t[0], "x": t[1], "y": t[2]}
        if len(t) >= 4 and t[3] is not None:
            m["coordinate_uncertainty"] = {"x": t[3][0], "y": t[3][1]}
        markers.append(m)
    return {
        "points": markers,
        "rows": 3,
        "cols": 3,
        "max_outliers": max_outliers,
        "tolerance": tolerance,
        "origin_bounds": VB3,
        "row_vector_bounds": VB3,
        "col_vector_bounds": VB3,
    }


def check_band_results(run):
    # 3x3 精确栅格 O=(0,0)、A=(2,0)、B=(0,2)，id 按行主序：
    # 第 r 行第 c 列坐标为 (2r, 2c)，id=2 的真值为 (0,2)。
    exact = [(r * 3 + c + 1, 2 * r, 2 * c) for r in range(3) for c in range(3)]

    # 场景一：中心报为 (1,2)、x 半宽 1 → 观测区间 [0,2] 覆盖真值。
    # 误差带内有效残差为 0：容差 0、零弃点即可复原全部 9 个标记。
    in_band = [
        (i, x + 1, y, (1, 0)) if i == 2 else (i, x, y)
        for i, x, y in exact
    ]
    r1 = run(_band_base_payload(in_band, tolerance=0, max_outliers=0))
    assert r1["solvable"] is True, r1.get("reason")
    p = r1["parameters"]
    assert (p["origin"], p["row_vector"], p["col_vector"]) == (
        [0, 0],
        [2, 0],
        [0, 2],
    )
    assert r1["objective"] == {
        "discarded_count": 0,
        "max_manhattan_residual": 0,
        "total_manhattan_residual": 0,
    }
    a2 = next(a for a in r1["assignments"] if a["id"] == 2)
    assert a2["predicted"] == [0, 2]
    assert a2["residual"] == [1, 0]
    assert a2["effective_residual"] == [0, 0]
    assert a2["manhattan_residual"] == 0
    print("  误差带内：越轴距离计 0，容差 0 下零弃点零残差复原")

    # 场景二：中心报为 (-2,2)、x 半宽 1 → 区间 [-3,-1]，真值越界 1。
    out_band = [
        (i, x - 2, y, (1, 0)) if i == 2 else (i, x, y)
        for i, x, y in exact
    ]
    # 2a：容差 0、不许弃点 → 明确无解（真实栅格冲突，区别于量测不确定度）
    r2 = run(_band_base_payload(out_band, tolerance=0, max_outliers=0))
    assert r2["solvable"] is False
    assert "容差" in r2["reason"], r2["reason"]
    # 2b：容差 0、允许 1 弃点 → id=2 被弃，证据有效 L∞ 恰为越界量 1
    r3 = run(_band_base_payload(out_band, tolerance=0, max_outliers=1))
    assert r3["solvable"] is True, r3.get("reason")
    assert r3["objective"]["discarded_count"] == 1
    d = r3["discarded"][0]
    assert d["id"] == 2
    assert d["coordinate_uncertainty"] == [1, 0]
    assert d["nearest_inf_residual"] == 1
    assert d["cells_within_tolerance"] == []
    # 2c：容差 1 → 越界量 1 被计入，最大/总和曼哈顿残差均为 1
    r4 = run(_band_base_payload(out_band, tolerance=1, max_outliers=0))
    assert r4["solvable"] is True, r4.get("reason")
    assert (
        r4["objective"]["discarded_count"],
        r4["objective"]["max_manhattan_residual"],
        r4["objective"]["total_manhattan_residual"],
    ) == (0, 1, 1)
    a2b = next(a for a in r4["assignments"] if a["id"] == 2)
    assert a2b["effective_residual"] == [1, 0]
    assert a2b["manhattan_residual"] == 1
    print("  误差带外：越界 1 计为有效残差；无解原因 / 弃点证据 / 放宽容差均正确")


def _run(payload):
    """有 BASE_URL 时走 HTTP，否则直测求解器。"""
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
            return json.loads(resp.read())
    points = []
    for m in payload["points"]:
        u = m.get("coordinate_uncertainty")
        if u is None:
            points.append((m["id"], m["x"], m["y"]))
        else:
            points.append((m["id"], m["x"], m["y"], (u["x"], u["y"])))
    return reconstruct(
        points,
        payload["rows"],
        payload["cols"],
        payload["tolerance"],
        payload["max_outliers"],
        BOUNDS,
    )


def main():
    points, _ = build_case()
    via = f"HTTP ({os.environ['BASE_URL']})" if os.environ.get("BASE_URL") else "求解器直测"
    print(f"==> 场景一：旧口径回归（漏读 + 划痕亮点 + 抖动），{via}")
    check_result(_run(expected_payload(points)))

    print(f"==> 场景二：轴向不确定度（误差带内零残差 / 带外超限），{via}")
    check_band_results(_run)

    print("==> 冒烟通过：旧口径回归与误差带场景均正确处理")


if __name__ == "__main__":
    main()
