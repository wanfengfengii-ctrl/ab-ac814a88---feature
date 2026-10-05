"""HTTP 层测试。"""

import os
import sys

from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app  # noqa: E402

client = TestClient(app)

VB = {"x": {"lo": -3, "hi": 3}, "y": {"lo": -3, "hi": 3}}


def payload(points, **over):
    base = {
        "points": [{"id": i, "x": x, "y": y} for i, x, y in points],
        "rows": 3,
        "cols": 3,
        "max_outliers": 0,
        "tolerance": 0,
        "origin_bounds": VB,
        "row_vector_bounds": VB,
        "col_vector_bounds": VB,
    }
    base.update(over)
    return base


def exact_points():
    return [
        (r * 3 + c + 1, 2 * r, 2 * c)
        for r in range(3)
        for c in range(3)
    ]


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_reconstruct_ok():
    r = client.post("/api/wafer-grids/reconstruct", json=payload(exact_points()))
    assert r.status_code == 200
    body = r.json()
    assert body["solvable"] is True
    assert body["parameters"]["origin"] == [0, 0]
    assert body["parameters"]["row_vector"] == [2, 0]
    assert body["parameters"]["col_vector"] == [0, 2]
    assert len({(a["row"], a["col"]) for a in body["assignments"]}) == 9
    for a in body["assignments"]:
        assert a["predicted"] == [a["x"], a["y"]]


def test_reconstruct_no_solution_body():
    pts = [(i, 100 + 3 * i, 200 + 3 * i) for i in range(1, 8)]
    r = client.post("/api/wafer-grids/reconstruct", json=payload(pts, max_outliers=2))
    assert r.status_code == 200
    body = r.json()
    assert body["solvable"] is False
    assert body["reason"]


def test_duplicate_ids_rejected():
    pts = [(1, 0, 0)] * 7
    r = client.post("/api/wafer-grids/reconstruct", json=payload(pts))
    assert r.status_code == 422


def test_interval_span_rejected():
    p = payload(exact_points())
    p["origin_bounds"] = {"x": {"lo": 0, "hi": 7}, "y": {"lo": 0, "hi": 0}}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    assert "跨度" in r.text


def test_counts_out_of_range():
    p = payload(exact_points())
    p["max_outliers"] = 3
    assert client.post("/api/wafer-grids/reconstruct", json=p).status_code == 422
    p = payload(exact_points())
    p["points"] = p["points"][:6]
    assert client.post("/api/wafer-grids/reconstruct", json=p).status_code == 422
    p = payload(exact_points())
    p["rows"] = 8
    assert client.post("/api/wafer-grids/reconstruct", json=p).status_code == 422


# ---------------------------------------------------------------------------
# coordinate_uncertainty（逐点轴向误差带）
# ---------------------------------------------------------------------------
def _err_loc(resp, axis):
    for err in resp.json()["detail"]:
        if err["loc"][-1] == axis:
            return err["loc"]
    return None


def test_uncertainty_invalid_x_locates_point_and_axis():
    p = payload(exact_points())
    p["points"][3]["coordinate_uncertainty"] = {"x": 4, "y": 0}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    loc = _err_loc(r, "x")
    assert loc is not None
    # loc 精确定位到第 4 个点（下标 3）的 x 轴半宽
    assert loc[:3] == ["body", "points", 3]
    assert "0~3" in r.text


def test_uncertainty_invalid_y_negative_locates_axis():
    p = payload(exact_points())
    p["points"][0]["coordinate_uncertainty"] = {"x": 0, "y": -1}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    loc = _err_loc(r, "y")
    assert loc == ["body", "points", 0, "coordinate_uncertainty", "y"]


def test_uncertainty_wrong_type_rejected():
    p = payload(exact_points())
    p["points"][1]["coordinate_uncertainty"] = {"x": 1.5, "y": 0}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    assert _err_loc(r, "x") is not None


def test_uncertainty_bool_rejected_with_axis_loc():
    # 布尔值不得被静默强转为 0/1
    p = payload(exact_points())
    p["points"][2]["coordinate_uncertainty"] = {"x": 0, "y": True}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    assert _err_loc(r, "y") == [
        "body",
        "points",
        2,
        "coordinate_uncertainty",
        "y",
    ]


def test_in_band_point_zero_residual_over_http():
    # id=1 中心 (1,0)、x 半宽 1 覆盖真格位 (0,0)；tolerance=0 下
    # 不带误差带无解，带误差带零残差复原。
    pts = exact_points()
    pts[0] = (1, 1, 0)
    r0 = client.post("/api/wafer-grids/reconstruct", json=payload(pts))
    assert r0.json()["solvable"] is False

    p = payload(pts)
    p["points"][0]["coordinate_uncertainty"] = {"x": 1, "y": 0}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 200
    body = r.json()
    assert body["solvable"] is True, body.get("reason")
    assert body["objective"] == {
        "discarded_count": 0,
        "max_manhattan_residual": 0,
        "total_manhattan_residual": 0,
    }
    a1 = next(a for a in body["assignments"] if a["id"] == 1)
    assert a1["predicted"] == [0, 0]
    assert a1["residual"] == [1, 0]
    assert a1["effective_residual"] == [0, 0]
    assert a1["manhattan_residual"] == 0
    assert a1["observation_interval"] == [[0, 2], [0, 0]]
    # 未提供误差带的点仍按中心口径：半宽显式回显为 0
    a2 = next(a for a in body["assignments"] if a["id"] == 2)
    assert a2["coordinate_uncertainty"] == [0, 0]


def test_out_of_band_exceedance_over_http():
    # id=1 中心 (2,0)、x 半宽 1：格位 (0,0) 在区间外 1 → 有效残差 1
    pts = exact_points()
    pts[0] = (1, 2, 0)
    p = payload(pts, tolerance=1)
    p["points"][0]["coordinate_uncertainty"] = {"x": 1, "y": 0}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    body = r.json()
    assert body["solvable"] is True, body.get("reason")
    obj = body["objective"]
    assert (
        obj["discarded_count"],
        obj["max_manhattan_residual"],
        obj["total_manhattan_residual"],
    ) == (0, 1, 1)
    a1 = next(a for a in body["assignments"] if a["id"] == 1)
    assert (a1["row"], a1["col"]) == (0, 0)
    assert a1["residual"] == [2, 0]
    assert a1["effective_residual"] == [1, 0]
    assert a1["manhattan_residual"] == 1


def test_real_conflict_distinct_from_uncertainty_over_http():
    # 合法请求但联合约束无法满足 → 200 + solvable=false + 明确原因
    pts = exact_points()
    pts[0] = (1, 3, 0)
    p = payload(pts)
    p["points"][0]["coordinate_uncertainty"] = {"x": 1, "y": 0}
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 200
    body = r.json()
    assert body["solvable"] is False
    assert body["reason"]
