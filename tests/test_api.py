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
# coordinate_uncertainty（轴向误差带）
# ---------------------------------------------------------------------------
def _band_payload(uncertainty_by_id, **over):
    p = payload(exact_points(), **over)
    for m in p["points"]:
        if m["id"] in uncertainty_by_id:
            m["coordinate_uncertainty"] = {
                "x": uncertainty_by_id[m["id"]][0],
                "y": uncertainty_by_id[m["id"]][1],
            }
    return p


def test_omitted_uncertainty_matches_legacy_response():
    r = client.post("/api/wafer-grids/reconstruct", json=payload(exact_points()))
    assert r.status_code == 200
    for a in r.json()["assignments"]:
        assert a["coordinate_uncertainty"] == [0, 0]
        assert a["effective_residual"] == a["residual"]
        assert a["manhattan_residual"] == sum(abs(v) for v in a["residual"])
    for d in [a for a in r.json()["assignments"] if not a["adopted"]]:
        assert d["coordinate_uncertainty"] == [0, 0]


def test_in_band_prediction_has_zero_residual():
    # id=2 真值 (0,2)，中心报为 (1,2)，x 半宽 1：容差 0 下零弃点、零残差
    p = _band_payload({2: (1, 0)})
    for m in p["points"]:
        if m["id"] == 2:
            m["x"] = 1
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["solvable"] is True
    assert body["objective"] == {
        "discarded_count": 0,
        "max_manhattan_residual": 0,
        "total_manhattan_residual": 0,
    }
    a2 = next(a for a in body["assignments"] if a["id"] == 2)
    assert a2["predicted"] == [0, 2]
    assert a2["residual"] == [1, 0]
    assert a2["effective_residual"] == [0, 0]
    assert a2["manhattan_residual"] == 0


def test_out_of_band_exceedance_counts_and_discards():
    # id=2 中心报 (-2,2)，x 半宽 1：越界 1。容差 0 → 无解；
    # 容差 0 + 允许 1 弃点 → 弃点证据有效 L∞ 为 1
    p = _band_payload({2: (1, 0)}, max_outliers=1)
    for m in p["points"]:
        if m["id"] == 2:
            m["x"] = -2
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 200
    body = r.json()
    assert body["solvable"] is True
    assert body["objective"]["discarded_count"] == 1
    d = body["discarded"][0]
    assert d["id"] == 2
    assert d["coordinate_uncertainty"] == [1, 0]
    assert d["nearest_inf_residual"] == 1
    assert d["cells_within_tolerance"] == []

    # 不容许弃点、容差 0 → 几何无解（真实栅格冲突而非请求非法）
    p0 = _band_payload({2: (1, 0)}, max_outliers=0)
    for m in p0["points"]:
        if m["id"] == 2:
            m["x"] = -2
    r0 = client.post("/api/wafer-grids/reconstruct", json=p0)
    assert r0.status_code == 200
    b0 = r0.json()
    assert b0["solvable"] is False
    assert "容差" in b0["reason"]

    # 容差 1 放宽 → 越界量计入，最大/总和曼哈顿残差均为 1
    p1 = _band_payload({2: (1, 0)}, tolerance=1, max_outliers=0)
    for m in p1["points"]:
        if m["id"] == 2:
            m["x"] = -2
    r1 = client.post("/api/wafer-grids/reconstruct", json=p1)
    assert r1.status_code == 200
    b1 = r1.json()
    assert b1["solvable"] is True
    assert (
        b1["objective"]["discarded_count"],
        b1["objective"]["max_manhattan_residual"],
        b1["objective"]["total_manhattan_residual"],
    ) == (0, 1, 1)


def test_illegal_halfwidth_points_to_marker_and_axis():
    # x 半宽 4（超过 3）：错误定位须到具体点的 x 轴
    p = _band_payload({2: (4, 0)})
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    locs = [tuple(e["loc"]) for e in r.json()["detail"]]
    assert any(
        loc[1:3] == ("points", 1)
        and "coordinate_uncertainty" in loc
        and loc[-1] == "x"
        for loc in locs
    ), locs

    # y 半宽为负
    p = _band_payload({5: (0, -1)})
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    locs = [tuple(e["loc"]) for e in r.json()["detail"]]
    assert any(
        loc[1:3] == ("points", 4)
        and "coordinate_uncertainty" in loc
        and loc[-1] == "y"
        for loc in locs
    ), locs

    # 半宽类型非法（非数字字符串；注意 "1" 会被 pydantic 宽松转为整数 1）
    p = _band_payload({2: (1, 0)})
    p["points"][1]["coordinate_uncertainty"]["x"] = "wide"
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    assert any(
        e["loc"][1:3] == ["points", 1] and e["loc"][-1] == "x"
        for e in r.json()["detail"]
    )


def test_uncertainty_must_be_object():
    p = payload(exact_points())
    p["points"][0]["coordinate_uncertainty"] = [1, 0]
    r = client.post("/api/wafer-grids/reconstruct", json=p)
    assert r.status_code == 422
    assert any(
        e["loc"][1:3] == ["points", 0]
        and e["loc"][-1] == "coordinate_uncertainty"
        for e in r.json()["detail"]
    )
