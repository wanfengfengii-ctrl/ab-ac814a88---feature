"""FastAPI 应用：晶圆标记栅格复原服务。"""

from typing import Optional

from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field, field_validator, model_validator

from .solver import reconstruct

app = FastAPI(
    title="Wafer Grid Reconstruction API",
    description=(
        "从带编号的无序整数坐标中恢复栅格原点、基向量与格位分配；"
        "支持逐点 x/y 轴向量测不确定度（误差带）。"
    ),
    version="1.1.0",
)


class CoordinateUncertainty(BaseModel):
    """单标记的轴向量测半宽：观测区间为 [x-hx, x+hx] × [y-hy, y+hy]。

    仅接受 0~3 的非负整数；非法值的 422 错误 loc 精确到
    ``points.<下标>.coordinate_uncertainty.<轴>``。
    """

    x: int = Field(..., description="x 轴向非负整数半宽（0~3，0 表示无误差带）")
    y: int = Field(..., description="y 轴向非负整数半宽（0~3，0 表示无误差带）")

    @field_validator("x", "y", mode="before")
    @classmethod
    def _check_half_width(cls, v, info):
        # pydantic 宽松模式会把 bool 静默强转为 int，须在强转前拦截；
        # 浮点/字符串等类型交给 pydantic 标准 int 校验
        if isinstance(v, bool):
            raise ValueError(
                f"{info.field_name} 轴半宽必须为 0~3 的非负整数（不能为布尔值）"
            )
        if isinstance(v, int) and (v < 0 or v > 3):
            raise ValueError(
                f"{info.field_name} 轴半宽 {v} 非法：必须为 0~3 的非负整数"
            )
        return v


class Marker(BaseModel):
    id: int = Field(..., description="标记唯一编号")
    x: int
    y: int
    coordinate_uncertainty: Optional[CoordinateUncertainty] = Field(
        default=None,
        description="选填；省略时 x、y 两轴半宽均视为 0（按标记中心计残差）",
    )


class Interval(BaseModel):
    """单个分量的闭区间 [lo, hi]，跨度不超过 6。"""

    lo: int
    hi: int

    @model_validator(mode="after")
    def _check(self):
        if self.hi < self.lo:
            raise ValueError(f"闭区间下界 {self.lo} 不得大于上界 {self.hi}")
        if self.hi - self.lo > 6:
            raise ValueError(
                f"区间 [{self.lo}, {self.hi}] 跨度 {self.hi - self.lo} 超过 6"
            )
        return self


class VectorBounds(BaseModel):
    x: Interval
    y: Interval


class ReconstructRequest(BaseModel):
    points: list[Marker] = Field(..., min_length=7, max_length=14)
    rows: int = Field(..., ge=3, le=7)
    cols: int = Field(..., ge=3, le=7)
    max_outliers: int = Field(..., ge=0, le=2)
    tolerance: int = Field(..., ge=0, description="逐分量（L∞）有效坐标容差")
    origin_bounds: VectorBounds
    row_vector_bounds: VectorBounds
    col_vector_bounds: VectorBounds

    @model_validator(mode="after")
    def _check_points(self):
        ids = [p.id for p in self.points]
        if len(set(ids)) != len(ids):
            raise ValueError("标记编号必须唯一")
        return self


def _bounds_pair(vb: VectorBounds):
    return ([vb.x.lo, vb.x.hi], [vb.y.lo, vb.y.hi])


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/api/wafer-grids/reconstruct")
async def reconstruct_grid(req: ReconstructRequest):
    # 按编号排序：字典序决胜项定义在“按编号排列的分配序列”上
    points = []
    for p in req.points:
        if p.coordinate_uncertainty is None:
            points.append((p.id, p.x, p.y))
        else:
            points.append(
                (p.id, p.x, p.y, (p.coordinate_uncertainty.x,
                                  p.coordinate_uncertainty.y))
            )
    points.sort(key=lambda t: t[0])
    bounds = {
        "origin": _bounds_pair(req.origin_bounds),
        "row_vector": _bounds_pair(req.row_vector_bounds),
        "col_vector": _bounds_pair(req.col_vector_bounds),
    }
    result = await run_in_threadpool(
        reconstruct,
        points,
        req.rows,
        req.cols,
        req.tolerance,
        req.max_outliers,
        bounds,
    )
    if not result.get("solvable"):
        # 几何上无解不是请求格式错误：以 200 返回明确的无解原因，
        # 由响应体 solvable=false 标识。
        return result
    return result
