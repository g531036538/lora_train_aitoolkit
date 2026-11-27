"""LoRAForge 本地界面（FastAPI + Jinja2，端口 7861）。

界面只做三件事：把库里的清单读出来给人看、把表单参数交给
``loraforge.configs``、把打包交给 ``loraforge.export``。
打标逻辑、YAML 结构、zip 结构都不在这一层 —— 底层换了界面这里不用重写。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from loraforge.caption.pipeline import BackendInfo, list_backends
from loraforge.caption.reasons import human_reason
from loraforge.configs import (
    VRAM_TIERS,
    build_config,
    default_plan,
    plan_for_tier,
    validate_config,
)
from loraforge.export import ExportError, package
from loraforge.models import PROFILES, DatasetKind, TargetModel

APP_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(title="LoRAForge", version="0.1.0", description="LoRA 打标与训练配置的本地界面")


# ---------------------------------------------------------------------------
# 数据装配：一律从 loraforge 读，界面不存任何参数默认值
# ---------------------------------------------------------------------------


def _backends() -> list[BackendInfo]:
    """单列一层，测试可替换成不可用后端来验渲染。"""
    return list_backends()


def model_cards() -> list[dict[str, Any]]:
    """三款模型 + 各自默认方案（默认值全部来自 ``default_plan``）。"""
    cards: list[dict[str, Any]] = []
    for target, prof in PROFILES.items():
        plan = default_plan(target)
        cards.append(
            {
                "key": target.value,
                "label": prof.label,
                "train_base": prof.train_base,
                "verify_base": prof.verify_base,
                "caption_style": prof.caption_style,
                "notes": list(prof.notes),
                "plan": plan,
            }
        )
    return cards


def backend_rows() -> list[dict[str, Any]]:
    """后端清单：不可用的也列出来，reason 翻成人话摆在行上。"""
    rows: list[dict[str, Any]] = []
    for info in _backends():
        rows.append(
            {
                "key": info.key,
                "label": info.label,
                "available": info.available,
                "reason": info.reason,
                "reason_text": human_reason(info.reason) if info.reason else "",
                "needs_gpu": info.caps.needs_gpu,
                "supports_tags": info.caps.supports_tags,
                "supports_prose": info.caps.supports_prose,
                "languages": list(info.caps.languages),
            }
        )
    return rows


def vram_rows() -> list[dict[str, Any]]:
    rows = [
        {
            "key": tier.key,
            "label": tier.label,
            "quantize": tier.quantize or "不量化",
            "batch_size": tier.batch_size,
            "max_resolution": tier.max_resolution,
            "note": tier.note,
        }
        for tier in VRAM_TIERS.values()
    ]
    return sorted(rows, key=lambda row: int(row["key"]))


def _fail(stage: str, errors: list[str]) -> JSONResponse:
    return JSONResponse({"ok": False, "stage": stage, "errors": errors}, status_code=400)


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "models": model_cards(),
            "backends": backend_rows(),
            "tiers": vram_rows(),
            "kinds": [kind.value for kind in DatasetKind],
            "default_model": TargetModel.SDXL.value,
            "default_tier": "24",
        },
    )


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    return {
        "ok": True,
        "models": [target.value for target in PROFILES],
        "backends": [info.key for info in _backends()],
    }


@app.post("/export")
def export_zip(
    dataset_dir: str = Form(...),
    out_zip: str = Form(...),
    model: str = Form(...),
    trigger: str = Form(""),
    vram_tier: str = Form("24"),
    include_images: str = Form("1"),
) -> JSONResponse:
    """按表单参数生成 yaml 并打包。硬错一律 400，缺标/孤儿以 200 + 报告返回。"""
    try:
        target = TargetModel(model)
    except ValueError:
        return _fail("参数", [f"未知目标模型：{model!r}"])

    dataset = Path(dataset_dir.strip()).expanduser()
    if not dataset.is_dir():
        return _fail("参数", [f"图集目录不存在：{dataset}"])

    trigger_text = trigger.strip()
    if not trigger_text:
        return _fail("参数", ["触发词不能为空：没有触发词，LoRA 学不进身份"])

    try:
        plan = plan_for_tier(
            target,
            vram_tier,
            dataset_dir=str(dataset),
            trigger=trigger_text,
        )
    except ValueError as exc:
        return _fail("参数", [str(exc)])

    config = build_config(target, plan)
    hard_errors = validate_config(config, target)
    if hard_errors:
        # 危险配置：不生成 yaml、不打包，直接把问题摆出来。
        return _fail("校验", hard_errors)

    out = Path(out_zip.strip()).expanduser()
    if out.suffix.lower() != ".zip":
        return _fail("参数", [f"导出路径要以 .zip 结尾：{out}"])

    out.parent.mkdir(parents=True, exist_ok=True)
    config_path = out.parent / "train.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    with_images = include_images.strip().lower() not in {"", "0", "false", "off"}
    try:
        report = package(dataset, config_path, out, include_images=with_images)
    except ExportError as exc:
        return _fail("导出", [str(exc)])

    return JSONResponse(
        {
            "ok": report.ok,
            "zip": str(report.zip_path),
            "config": str(config_path),
            "images": report.images,
            "captions": report.captions,
            "missing_caption": report.missing_caption,
            "orphan_caption": report.orphan_caption,
            "ambiguous": report.ambiguous,
            "with_images": with_images,
        }
    )


__all__ = ["app", "backend_rows", "model_cards", "vram_rows"]
