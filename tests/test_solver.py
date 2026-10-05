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


def with_band(pts, overrides):
    """把 {id: (hx, hy)} 写入点列（4 元组）。"""
    out = []
    for t in pts:
        if t[0] in overrides:
            out.append((t[0], t[1], t[2], overrides[t[0]]))
        else:
            out.append(t)
    return out



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
# 轴向量测不确定度（误差带）
# ---------------------------------------------------------------------------
def test_zero_band_is_identical_to_no_band():
    pts = grid_points(3, 3, (0, 0), (2, 0), (0, 2))
    pts[0] = (1, 1, 0)  # 制造一个非精确点，确保残差字段确实参与比较
    pts_band = [(i, x, y, (0, 0)) for i, x, y in pts]
    r1 = reconstruct(pts, 3, 3, 1, 1, BOUNDS_WIDE)
    r2 = reconstruct(pts_band, 3, 3, 1, 1, BOUNDS_WIDE)
    # 半宽全 0 时所有既有字段必须逐值一致
    for key in ("objective", "parameters", "assignments", "discarded"):
        assert r1[key] == r2[key]


def test_in_band_prediction_has_zero_effective_residual():
    # 真格位 (0,0)；低信噪下报告中心偏到 (1,0)，x 半宽 1 覆盖真格位。
    # tolerance=0：不提供误差带时该点无容差可达格位；提供后有效残差为 0。
    pts = grid_points(3, 3, (0, 0), (2, 0), (0, 2))
    pts[0] = (1, 1, 0)
    no_band = reconstruct(pts, 3, 3, 0, 0, BOUNDS_WIDE)
    assert no_band["solvable"] is False

    banded = reconstruct(with_band(pts, {1: (1, 0)}), 3, 3, 0, 0, BOUNDS_WIDE)
    assert banded["solvable"] is True, banded.get("reason")
    obj = banded["objective"]
    assert (
        obj["discarded_count"],
        obj["max_manhattan_residual"],
        obj["total_manhattan_residual"],
    ) == (0, 0, 0)
    a1 = next(a for a in banded["assignments"] if a["id"] == 1)
    assert a1["adopted"] is True
    assert (a1["row"], a1["col"]) == (0, 0)
    assert a1["predicted"] == [0, 0]
    # 原始偏差仍按标记中心报告；有效残差在误差带内归零
    assert a1["residual"] == [1, 0]
    assert a1["effective_residual"] == [0, 0]
    assert a1["manhattan_residual"] == 0
    assert a1["observation_interval"] == [[0, 2], [0, 0]]
    p = banded["parameters"]
    assert (p["origin"], p["row_vector"], p["col_vector"]) == (
        [0, 0],
        [2, 0],
        [0, 2],
    )


def test_single_axis_band_covers_only_that_axis():
    pts = grid_points(3, 3, (0, 0), (2, 0), (0, 2))
    # y 方向偏 1：仅给 y 半宽 → 有效残差 0；给 x 半宽则 y 仍越界
    pts[0] = (1, 0, 1)
    hy = reconstruct(with_band(pts, {1: (0, 1)}), 3, 3, 0, 0, BOUNDS_WIDE)
    assert hy["solvable"] is True
    a1 = next(a for a in hy["assignments"] if a["id"] == 1)
    assert a1["effective_residual"] == [0, 0]
    hx = reconstruct(with_band(pts, {1: (1, 0)}), 3, 3, 0, 0, BOUNDS_WIDE)
    assert hx["solvable"] is False


def test_out_of_band_distance_counts_into_objectives():
    # id=1 真格位 (0,0)，报告中心 (2,0)、x 半宽 1（区间 [1,3]）：
    # 格位 (0,0) 在区间外 1 个单位 → 有效残差 1，计入最大残差与残差和；
    # 而吸附到精确格位 (2,0) 会挤掉 id=2，互异格位裁决不允许。
    pts = grid_points(3, 3, (0, 0), (2, 0), (0, 2))
    pts[0] = (1, 2, 0)
    res = reconstruct(with_band(pts, {1: (1, 0)}), 3, 3, 1, 0, BOUNDS_WIDE)
    assert res["solvable"] is True, res.get("reason")
    obj = res["objective"]
    assert (
        obj["discarded_count"],
        obj["max_manhattan_residual"],
        obj["total_manhattan_residual"],
    ) == (0, 1, 1)
    by_id = {a["id"]: a for a in res["assignments"]}
    assert (by_id[1]["row"], by_id[1]["col"]) == (0, 0)
    assert by_id[1]["residual"] == [2, 0]
    assert by_id[1]["effective_residual"] == [1, 0]
    assert (by_id[2]["row"], by_id[2]["col"]) == (0, 1)
    # 目标值与逐点有效残差同一口径
    adopted = [a for a in res["assignments"] if a["adopted"]]
    assert max(a["manhattan_residual"] for a in adopted) == obj[
        "max_manhattan_residual"
    ]
    assert sum(a["manhattan_residual"] for a in adopted) == obj[
        "total_manhattan_residual"
    ]


def test_beyond_band_plus_tolerance_is_real_conflict():
    # tolerance=0 时其余 8 个精确点唯一钉住真栅格。id=1 中心 (3,0)：
    # 半宽 1（区间 [2,4]）只覆盖格位 (2,0)，而该格位属于精确点 id=2，
    # 互异格位 + 禁止弃点 → 明确无解（真实栅格冲突，而非量测不确定度）；
    # 半宽放宽到 3（区间 [0,6] 覆盖空格位 (0,0)）即零残差采用。
    pts = grid_points(3, 3, (0, 0), (2, 0), (0, 2))
    pts[0] = (1, 3, 0)
    tight = reconstruct(with_band(pts, {1: (1, 0)}), 3, 3, 0, 0, BOUNDS_WIDE)
    assert tight["solvable"] is False
    assert "容差" in tight["reason"]
    wide = reconstruct(with_band(pts, {1: (3, 0)}), 3, 3, 0, 0, BOUNDS_WIDE)
    assert wide["solvable"] is True
    a1 = next(a for a in wide["assignments"] if a["id"] == 1)
    assert (a1["row"], a1["col"]) == (0, 0)
    assert a1["effective_residual"] == [0, 0]


def test_band_discarded_evidence_uses_effective_distance():
    # 误差带扩到极限仍够不到任何格位的划痕亮点：证据距离按有效口径，
    # cells_within_tolerance 为空。
    pts = grid_points(3, 3, (0, 0), (2, 0), (0, 2))
    pts.append((42, 13, 13))
    res = reconstruct(
        with_band(pts, {42: (3, 3)}), 3, 3, 1, 1, BOUNDS_WIDE
    )
    assert res["solvable"] is True
    d = next(x for x in res["discarded"] if x["id"] == 42)
    assert d["coordinate_uncertainty"] == [3, 3]
    assert d["nearest_inf_residual"] == 13 - 4 - 3  # 最近格位 (4,4)
    assert d["nearest_manhattan_residual"] == 2 * (13 - 4 - 3)
    assert d["cells_within_tolerance"] == []
    assert "无可达格位" in d["reason"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
