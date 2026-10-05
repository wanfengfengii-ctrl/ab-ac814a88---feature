"""核心求解器测试。"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.solver import reconstruct  # noqa: E402

BOUNDS_WIDE = {
    "origin": ([-3, 3], [-3, 3]),
    "row_vector": ([-3, 3], [-3, 3]),
    "col_vector": ([-3, 3], [-3, 3]),
}


def grid_points(rows, cols, origin, av, bv):
    ox, oy = origin
    ax, ay = av
    bx, by = bv
    pts = []
    pid = 1
    for r in range(rows):
        for c in range(cols):
            pts.append((pid, ox + r * ax + c * bx, oy + r * ay + c * by))
            pid += 1
    return pts


def test_exact_grid_recovers_parameters():
    pts = grid_points(3, 3, (0, 0), (2, 1), (-1, 2))
    res = reconstruct(pts, 3, 3, 0, 0, BOUNDS_WIDE)
    assert res["solvable"] is True, res.get("reason")
    p = res["parameters"]
    assert (p["origin"], p["row_vector"], p["col_vector"]) == (
        [0, 0],
        [2, 1],
        [-1, 2],
    )
    assert p["determinant"] == 5
    obj = res["objective"]
    assert (
        obj["discarded_count"],
        obj["max_manhattan_residual"],
        obj["total_manhattan_residual"],
    ) == (0, 0, 0)
    adopted = [a for a in res["assignments"] if a["adopted"]]
    cells = [(a["row"], a["col"]) for a in adopted]
    assert len(cells) == len(set(cells)) == 9


def test_missing_markers_and_scratch_outliers():
    # 4x4，漏读 4 格 + 2 划痕亮点 + 抖动
    pts, _ = __import__("scripts.smoke", fromlist=["build_case"]).build_case()
    res = reconstruct(pts, 4, 4, 1, 2, BOUNDS_WIDE)
    assert res["solvable"] is True, res.get("reason")
    p = res["parameters"]
    assert p["origin"] == [0, 0]
    assert p["row_vector"] == [3, 0]
    assert p["col_vector"] == [0, 3]
    obj = res["objective"]
    assert (
        obj["discarded_count"],
        obj["max_manhattan_residual"],
        obj["total_manhattan_residual"],
    ) == (2, 1, 3)
    discarded = {a["id"] for a in res["assignments"] if not a["adopted"]}
    assert discarded == {90, 91}
    adopted = [a for a in res["assignments"] if a["adopted"]]
    assert len(adopted) == 12
    cells = [(a["row"], a["col"]) for a in adopted]
    assert len(set(cells)) == 12
    for d in res["discarded"]:
        assert d["cells_within_tolerance"] == []
        assert d["nearest_inf_residual"] > 1


def test_unsolvable_when_points_far_away():
    pts = [(i, 100 + 3 * i, 200 + 3 * i) for i in range(1, 8)]
    res = reconstruct(pts, 3, 3, 0, 2, BOUNDS_WIDE)
    assert res["solvable"] is False
    assert "容差" in res["reason"]


def test_unsolvable_with_tight_origin_bounds():
    pts = grid_points(3, 3, (5, 5), (2, 0), (0, 2))
    tight = {
        "origin": ([-1, 1], [-1, 1]),
        "row_vector": ([1, 3], [-1, 1]),
        "col_vector": ([-1, 1], [1, 3]),
    }
    res = reconstruct(pts, 3, 3, 0, 2, tight)
    assert res["solvable"] is False
    assert res["reason"]


def test_negative_determinant_basis_rejected():
    # A=(0,2), B=(2,0) 的 det=-4（错误手性）；正确解 A=(2,0),B=(0,2)
    pts = grid_points(3, 3, (0, 0), (2, 0), (0, 2))
    res = reconstruct(pts, 3, 3, 0, 0, BOUNDS_WIDE)
    assert res["solvable"] is True
    p = res["parameters"]
    assert p["row_vector"] == [2, 0]
    assert p["col_vector"] == [0, 2]
    assert p["determinant"] > 0


def test_outlier_cap_zero_forces_no_discard():
    # 7 个点：6 个恰好落在 3x3 栅格上，1 个划痕；max_outliers=0 → 无解
    pts = grid_points(3, 3, (0, 0), (2, 0), (0, 2))[:6]
    pts.append((42, 13, 13))
    res_no = reconstruct(pts, 3, 3, 0, 0, BOUNDS_WIDE)
    assert res_no["solvable"] is False
    res_yes = reconstruct(pts, 3, 3, 0, 1, BOUNDS_WIDE)
    assert res_yes["solvable"] is True
    assert res_yes["objective"]["discarded_count"] == 1
    d = res_yes["discarded"][0]
    assert d["id"] == 42
    assert d["nearest_inf_residual"] > 0


def test_no_two_markers_share_cell_under_tolerance():
    # 两个标记都落在格位 (0,0) 的容差邻域内；3x3 其余格位放精确点。
    pts = [(1, 0, 0), (2, 1, 0)]  # 第二个只能容差吸附到 (0,0) 或 (2,0)
    pid = 3
    for r in range(3):
        for c in range(3):
            if (r, c) == (0, 0):
                continue
            pts.append((pid, 2 * r, 2 * c))
            pid += 1
    res = reconstruct(pts, 3, 3, 1, 2, BOUNDS_WIDE)
    assert res["solvable"] is True, res.get("reason")
    adopted = [a for a in res["assignments"] if a["adopted"]]
    cells = [(a["row"], a["col"]) for a in adopted]
    assert len(cells) == len(set(cells))
    by_id = {a["id"]: a for a in adopted}
    # id=2 仅能与 id=1 争 (0,0) 或与精确点争 (1,0)：它只能被弃点，
    # 绝不能与任何标记共用格位
    assert res["objective"]["discarded_count"] == 1
    assert 2 not in by_id
    assert 1 in by_id and (by_id[1]["row"], by_id[1]["col"]) == (0, 0)


def test_unordered_input_and_ids_define_output_order():
    pts = grid_points(3, 4, (1, -1), (3, 0), (0, 3))
    import random

    random.Random(7).shuffle(pts)
    res = reconstruct(pts, 3, 4, 0, 0, BOUNDS_WIDE)
    assert res["solvable"] is True, res.get("reason")
    assert [a["id"] for a in res["assignments"]] == sorted(
        a["id"] for a in res["assignments"]
    )
    p = res["parameters"]
    assert (p["origin"], p["row_vector"], p["col_vector"]) == (
        [1, -1],
        [3, 0],
        [0, 3],
    )
    assert p["determinant"] == 9


def test_residual_componentwise_within_tolerance():
    pts, _ = __import__("scripts.smoke", fromlist=["build_case"]).build_case()
    res = reconstruct(pts, 4, 4, 1, 2, BOUNDS_WIDE)
    for a in res["assignments"]:
        if a["adopted"]:
            assert abs(a["residual"][0]) <= 1
            assert abs(a["residual"][1]) <= 1


# ---------------------------------------------------------------------------
# 轴向不确定度（误差带）
# ---------------------------------------------------------------------------
def _exact_grid_3x3():
    return grid_points(3, 3, (0, 0), (2, 0), (0, 2))


def test_point_inside_band_has_zero_effective_residual():
    # 格位 (0,1) 真值 (0,2)；设备只给出中心 (1,2)、x 半宽 1，
    # 观测区间 [0,2] 覆盖真值 → 有效残差为 0，容差 0 也应零弃点复原。
    pts = [
        (pid, x + 1 if pid == 2 else x, y, (1, 0) if pid == 2 else None)
        for pid, x, y in _exact_grid_3x3()
    ]
    res = reconstruct(pts, 3, 3, 0, 0, BOUNDS_WIDE)
    assert res["solvable"] is True, res.get("reason")
    p = res["parameters"]
    assert (p["origin"], p["row_vector"], p["col_vector"]) == (
        [0, 0],
        [2, 0],
        [0, 2],
    )
    assert res["objective"] == {
        "discarded_count": 0,
        "max_manhattan_residual": 0,
        "total_manhattan_residual": 0,
    }
    a2 = next(a for a in res["assignments"] if a["id"] == 2)
    assert a2["adopted"] is True
    assert (a2["row"], a2["col"]) == (0, 1)
    assert a2["predicted"] == [0, 2]
    # 原始残差仍相对观测中心 (1,2)；有效残差扣除误差带后为 0
    assert a2["residual"] == [1, 0]
    assert a2["effective_residual"] == [0, 0]
    assert a2["manhattan_residual"] == 0


def test_band_rescues_legal_marker_that_center_value_would_discard():
    # 同样的中心 (1,2)：不给误差带、容差 0 时必须弃点；
    # 给 x 半宽 1 后被合法采用，且不占弃点名额。
    pts = [
        (pid, x + 1 if pid == 2 else x, y)
        for pid, x, y in _exact_grid_3x3()
    ]
    pts_band = [
        (pid, x, y, (1, 0) if pid == 2 else None) if pid == 2 else (pid, x, y)
        for pid, x, y in pts
    ]
    res_flat = reconstruct(pts, 3, 3, 0, 0, BOUNDS_WIDE)
    assert res_flat["solvable"] is False
    res_band = reconstruct(pts_band, 3, 3, 0, 0, BOUNDS_WIDE)
    assert res_band["solvable"] is True
    assert res_band["objective"]["discarded_count"] == 0
    assert {a["id"] for a in res_band["assignments"] if a["adopted"]} == set(
        range(1, 10)
    )


def test_out_of_band_distance_counts_toward_tolerance():
    # 中心 (-2,2)、x 半宽 1 → 区间 [-3,-1]；真值 (0,2) 越界 1。
    pts = [
        (pid, x - 2 if pid == 2 else x, y, (1, 0) if pid == 2 else None)
        for pid, x, y in _exact_grid_3x3()
    ]
    # 容差 0、不许弃点：越界 1 超限 → 无解
    res0 = reconstruct(pts, 3, 3, 0, 0, BOUNDS_WIDE)
    assert res0["solvable"] is False
    # 容差 1：越界距离 1 被计入 → 可采用，有效曼哈顿残差为 1
    res1 = reconstruct(pts, 3, 3, 1, 0, BOUNDS_WIDE)
    assert res1["solvable"] is True, res1.get("reason")
    obj = res1["objective"]
    assert (
        obj["discarded_count"],
        obj["max_manhattan_residual"],
        obj["total_manhattan_residual"],
    ) == (0, 1, 1)
    a2 = next(a for a in res1["assignments"] if a["id"] == 2)
    assert (a2["row"], a2["col"]) == (0, 1)
    assert a2["effective_residual"] == [1, 0]
    assert a2["manhattan_residual"] == 1
    # 容差 0、允许 1 个弃点：弃点证据按误差带口径报告最近残差
    resd = reconstruct(pts, 3, 3, 0, 1, BOUNDS_WIDE)
    assert resd["solvable"] is True
    assert resd["objective"]["discarded_count"] == 1
    d = resd["discarded"][0]
    assert d["id"] == 2
    assert d["coordinate_uncertainty"] == [1, 0]
    assert d["nearest_inf_residual"] == 1
    assert d["cells_within_tolerance"] == []


def test_per_axis_bands_independent():
    # 中心 (1,3)，hx=1（覆盖 x=0）、hy=0；真值 (0,2)：x 有效残差 0、y 为 1
    pts = [
        (pid, x + 1, y + 1, (1, 0)) if pid == 2 else (pid, x, y)
        for pid, x, y in _exact_grid_3x3()
    ]
    res0 = reconstruct(pts, 3, 3, 0, 1, BOUNDS_WIDE)
    assert res0["solvable"] is True
    assert {d["id"] for d in res0["discarded"]} == {2}
    pts2 = [
        (t[0], t[1], t[2], (1, 1)) if t[0] == 2 else t
        for t in pts
    ]
    res1 = reconstruct(pts2, 3, 3, 0, 0, BOUNDS_WIDE)
    assert res1["solvable"] is True
    a2 = next(a for a in res1["assignments"] if a["id"] == 2)
    assert a2["effective_residual"] == [0, 0]


def test_band_far_points_still_unsolvable():
    # 真实栅格冲突不会被 0–3 的半宽掩盖：区间仍远离任何可行格位
    pts = [(i, 100 + 3 * i, 200 + 3 * i, (3, 3)) for i in range(1, 8)]
    res = reconstruct(pts, 3, 3, 0, 2, BOUNDS_WIDE)
    assert res["solvable"] is False
    assert "容差" in res["reason"]
    # 无解原因须说明已计入误差带，帮助区分量测不确定度与真实栅格冲突
    assert "不确定度" in res["reason"]


def test_zero_halfwidths_match_legacy_caliber():
    pts = [
        (pid, x, y, (0, 0))
        for pid, x, y in _exact_grid_3x3()
    ]
    res = reconstruct(pts, 3, 3, 0, 0, BOUNDS_WIDE)
    assert res["solvable"] is True
    for a in res["assignments"]:
        assert a["coordinate_uncertainty"] == [0, 0]
        assert a["effective_residual"] == a["residual"]
        assert a["manhattan_residual"] == sum(map(abs, a["residual"]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
