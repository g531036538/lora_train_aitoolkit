"""LoRAForge 本地界面（FastAPI + Jinja2，端口 7861）。

界面只做三件事：把库里的清单读出来给人看、把表单参数交给
``loraforge.configs``、把打包交给 ``loraforge.export``。
打标逻辑、YAML 结构、zip 结构都不在这一层 —— 底层换了界面这里不用重写。
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from loraforge.caption.base import CaptionBackend, CaptionContext
from loraforge.caption.pipeline import BackendInfo, get_backend, list_backends, run_batch
from loraforge.caption.reasons import human_reason
from loraforge.configs import (
    VRAM_TIERS,
    build_config,
    default_plan,
    plan_for_tier,
    validate_config,
)
from loraforge.export import ExportError, package, scan_images
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
# 打标后端实例：取一次，整批复用
# ---------------------------------------------------------------------------

_BACKEND_LOCK = threading.Lock()
#: ``key -> 后端实例``。wd_tagger 首次建 session 约 2 s，之后 0.78 s/张；
#: 每张图重新 ``get_backend`` 会把这 2 s 反复付一遍。
_BACKEND_CACHE: dict[str, CaptionBackend] = {}


def _caption_backend(key: str) -> CaptionBackend:
    """按 key 取后端（带缓存）。未知 key 抛 ``KeyError``，不可用抛 ``RuntimeError``。"""
    normalized = str(key).strip().lower() or "wd_tagger"
    with _BACKEND_LOCK:
        backend = _BACKEND_CACHE.get(normalized)
        if backend is None:
            backend = get_backend(normalized)
            _BACKEND_CACHE[normalized] = backend
        return backend


def _forget_backends() -> None:
    """清掉后端缓存：换模型 / 权重换新后调用。"""
    with _BACKEND_LOCK:
        _BACKEND_CACHE.clear()


def _wanted_images(raw: str, available: list[str]) -> list[str]:
    """``images`` 字段（换行或逗号分隔）→ 要跑的图名。空 = 全部。

    只重跑失败的几张就靠它：前端把失败图名回填进来。
    """
    names = [part.strip() for part in raw.replace("\r", "\n").replace(",", "\n").split("\n")]
    chosen = [name for name in names if name]
    if not chosen:
        return list(available)
    unknown = [name for name in chosen if name not in available]
    if unknown:
        raise ValueError(f"目录里没有这些图：{', '.join(unknown)}")
    return chosen


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


@app.post("/scan")
def scan(dataset_dir: str = Form(...)) -> JSONResponse:
    """清点图集：哪些是训练图、哪张已有 .txt、哪些文件被跳过。"""
    raw = dataset_dir.strip()
    if not raw:
        return _fail("参数", ["图集目录不能为空"])
    try:
        result = scan_images(Path(raw).expanduser())
    except ExportError as exc:
        return _fail("参数", [str(exc)])

    payload: dict[str, Any] = {
        "ok": True,
        "root": str(result.root),
        "count": len(result.images),
        "with_caption": len(result.with_caption),
        "images": [
            {
                "name": entry.name,
                "size": entry.size,
                "has_caption": entry.has_caption,
                "caption_bytes": entry.caption_bytes,
            }
            for entry in result.images
        ],
        "skipped": {"count": len(result.skipped), "names": result.skipped},
    }
    if not result.images:
        payload["warning"] = "目录里没有 .jpg/.jpeg/.png，打标无从跑起"
    return JSONResponse(payload)


@app.post("/caption")
def caption(
    dataset_dir: str = Form(...),
    model: str = Form(...),
    kind: str = Form(...),
    trigger: str = Form(""),
    backend: str = Form("wd_tagger"),
    overwrite: str = Form("1"),
    images: str = Form(""),
    refresh: str = Form("0"),
) -> JSONResponse:
    """对图集跑一批打标。单张失败不中断，失败原因原样带回。"""
    try:
        target = TargetModel(model)
    except ValueError:
        return _fail("参数", [f"未知目标模型：{model!r}"])
    try:
        dataset_kind = DatasetKind(kind)
    except ValueError:
        return _fail("参数", [f"未知数据集类型：{kind!r}"])

    raw = dataset_dir.strip()
    if not raw:
        return _fail("参数", ["图集目录不能为空"])
    try:
        scan_result = scan_images(Path(raw).expanduser())
    except ExportError as exc:
        return _fail("参数", [str(exc)])
    if not scan_result.images:
        return _fail("参数", [f"目录里没有可打标的图（只认 .jpg/.jpeg/.png）：{scan_result.root}"])

    available = [entry.name for entry in scan_result.images]
    try:
        wanted = _wanted_images(images, available)
    except ValueError as exc:
        return _fail("参数", [str(exc)])

    # 换了模型权重就重建后端：否则会拿旧 session 去打新图
    if refresh.strip().lower() not in {"", "0", "false", "off"}:
        _forget_backends()

    try:
        backend_obj = _caption_backend(backend)
    except KeyError as exc:
        return _fail("后端", [f"未知后端：{backend!r}（{exc}）"])
    except RuntimeError as exc:
        # 后端自己给出的 reason，原样返回，不自己另造一句
        return _fail("后端", [str(exc)])

    overwrite_existing = overwrite.strip().lower() not in {"", "0", "false", "off"}
    paths = [
        entry.path
        for entry in scan_result.images
        if entry.name in set(wanted)
    ]
    # 只有「不覆盖」时，已有 .txt 才算 skipped；覆盖时那是被重写，不是跳过
    before = (
        set()
        if overwrite_existing
        else {path.name for path in paths if path.with_suffix(".txt").is_file()}
    )
    ctx = CaptionContext(
        target=target,
        kind=dataset_kind,
        trigger=trigger.strip(),
    )
    try:
        report = run_batch(
            paths,
            ctx,
            backend_obj,
            overwrite=overwrite_existing,
        )
    except Exception as exc:  # 兜底：批量级别的失败也要说清楚，不吞
        return _fail("打标", [f"{type(exc).__name__}: {exc}"])

    items: list[dict[str, Any]] = []
    failed_items: list[dict[str, Any]] = []
    for item in report.items:
        if item.error is not None:
            status = "failed"
            failed_items.append({"image": item.image.name, "error": item.error, "retryable": True})
        elif item.image.name in before:
            status = "skipped"
        else:
            status = "written"
        items.append(
            {
                "image": item.image.name,
                "status": status,
                "caption": item.caption,
                "error": item.error,
            }
        )

    return JSONResponse(
        {
            "ok": report.failed == 0,
            "backend": getattr(backend_obj, "name", backend),
            "total": len(paths),
            "written": report.written,
            "failed": report.failed,
            "skipped": report.skipped,
            "items": items,
            "failed_items": failed_items,
        }
    )


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
