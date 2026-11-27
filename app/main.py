"""LoRAForge 本地界面（FastAPI + Jinja2，端口 7861）。

界面只做三件事：把库里的清单读出来给人看、把表单参数交给
``loraforge.configs``、把打包交给 ``loraforge.export``。
打标逻辑、YAML 结构、zip 结构都不在这一层 —— 底层换了界面这里不用重写。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Form, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from loraforge.caption.base import CaptionBackend, CaptionContext
from loraforge.caption.checks import CheckResult, check_captions
from loraforge.caption.pipeline import BackendInfo, get_backend, list_backends, run_batch
from loraforge.caption.reasons import human_reason
from loraforge.configs import (
    VRAM_TIERS,
    build_config,
    default_plan,
    plan_for_tier,
    render_acceptance_markdown,
    validate_config_detailed,
)
from loraforge.export import ExportError, package, scan_images
from loraforge.models import PROFILES, DatasetKind, TargetModel

from .markdown import render_markdown

APP_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))

app = FastAPI(title="LoRAForge", version="0.1.0", description="LoRA 打标与训练配置的本地界面")

#: 界面配置存盘格式版本。字段增减时 +1，旧文件按「版本不匹配」拒绝，不猜。
STATE_SCHEMA = 1
#: 存盘位置可用环境变量覆盖（测试用；正常用户不该设它）。
STATE_ENV = "LORAFORGE_UI_STATE"
#: 这个容量以下开量化，量化 + 全层 offload 同时上会伤导出 LoRA 的推理质量。
OFFLOAD_MAX_GB = 12
OFFLOAD_HINT = "量化 + 全层 offload 同时开，已有报告训练正常但导出的 LoRA 推理无效"
#: 校对页上色的唯一依据；界面不按 code 白名单，新 code 只要带了 severity 就能显示。
SEVERITY_LABEL = {"error": "红", "warn": "黄", "info": "正常"}

#: 愿意被存进 ui_state.json 的字段：只存表单值，不存图片、路径清单、后端实例。
STATE_FIELDS = (
    "model",
    "vram_tier",
    "kind",
    "trigger",
    "dataset_dir",
    "backend",
    "out_zip",
    "steps",
    "rank",
    "alpha",
    "lr",
    "batch_size",
    "include_images",
    "overwrite",
    "refresh",
)


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
            # 小显存档同时开了量化和 offload：这句是给人看的后果，不只是参数名
            "offload_hint": (
                OFFLOAD_HINT
                if tier.quantize and int(tier.key) <= OFFLOAD_MAX_GB
                else ""
            ),
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
# 配置存盘（~/.loraforge/ui_state.json）
# ---------------------------------------------------------------------------


def state_path() -> Path:
    """配置存盘位置：默认用户主目录，测试用 ``LORAFORGE_UI_STATE`` 指到 .scratch。"""
    override = os.environ.get(STATE_ENV, "").strip()
    if override:
        return Path(override).expanduser()
    return Path.home() / ".loraforge" / "ui_state.json"


def load_state() -> dict[str, Any]:
    """读回上次保存的表单值。读不到 / 版本不对都如实说，不静默当成空配置。"""
    path = state_path()
    if not path.is_file():
        return {"ok": False, "errors": [f"还没有保存过配置（{path} 不存在）"], "values": {}}
    try:
        payload = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return {"ok": False, "errors": [f"配置读不出来：{exc}"], "values": {}}
    if not isinstance(payload, dict):
        return {"ok": False, "errors": ["配置文件不是 JSON 对象"], "values": {}}
    schema = payload.get("schema")
    if schema != STATE_SCHEMA:
        return {
            "ok": False,
            "schema": schema,
            "errors": [f"配置版本不匹配：文件 schema={schema!r}，界面只认 {STATE_SCHEMA}"],
            "values": {},
        }
    values = payload.get("values") or {}
    kept = {
        key: value
        for key, value in values.items()
        if key in STATE_FIELDS and isinstance(value, str)
    }
    return {
        "ok": True,
        "schema": STATE_SCHEMA,
        "saved_at": payload.get("saved_at"),
        "values": kept,
    }


def _write_state(values: dict[str, Any]) -> dict[str, Any]:
    path = state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": STATE_SCHEMA,
        "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "values": values,
    }
    body = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    path.write_bytes(body.encode("utf-8"))
    return payload


# ---------------------------------------------------------------------------
# 校对 / 参数预览：装配数据，判断全部交给 loraforge
# ---------------------------------------------------------------------------


def _caption_text(path: Path) -> str:
    """读同名 .txt；没有就是空打标词（不是「跳过」）。"""
    txt = path.with_suffix(".txt")
    if not txt.is_file():
        return ""
    return txt.read_bytes().decode("utf-8", errors="replace")


def review_items(dataset_dir: str, target: TargetModel, ctx: CaptionContext) -> list[dict[str, Any]]:
    """逐图结果：caption + 句柄 severity + message 原文，界面只按 severity 上色。"""
    result = scan_images(Path(dataset_dir).expanduser())
    pairs = [(entry.path, _caption_text(entry.path)) for entry in result.images]
    checks = {check.image: check for check in check_captions(pairs, ctx)}
    items: list[dict[str, Any]] = []
    for entry in result.images:
        check: CheckResult = checks[entry.path]
        items.append(
            {
                "image": entry.name,
                "caption": _caption_text(entry.path),
                "has_caption": entry.has_caption,
                "code": check.code,
                "severity": check.severity,
                "severity_label": SEVERITY_LABEL.get(check.severity, check.severity),
                "message": check.message,
            }
        )
    return items


def severity_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"error": 0, "warn": 0, "info": 0}
    for item in items:
        counts[item["severity"]] = counts.get(item["severity"], 0) + 1
    return counts


def _parse_number(raw: str, label: str, cast: type) -> tuple[Any, str | None]:
    text = str(raw).strip()
    if not text:
        return None, None
    try:
        value = cast(text)
    except ValueError:
        return None, f"{label}不是数字：{raw!r}"
    return value, None


def build_plan_from_form(
    target: TargetModel,
    vram_tier: str,
    *,
    dataset_dir: str = "",
    trigger: str = "",
    output_dir: str = "",
    steps: str = "",
    rank: str = "",
    alpha: str = "",
    lr: str = "",
    batch_size: str = "",
) -> tuple[dict[str, Any], list[str]]:
    """表单 → ``plan_for_tier``；空字段不传，默认值由库里给，界面不填数。"""
    overrides: dict[str, Any] = {}
    errors: list[str] = []
    if dataset_dir.strip():
        overrides["dataset_dir"] = dataset_dir.strip()
    if trigger.strip():
        overrides["trigger"] = trigger.strip()
    if output_dir.strip():
        overrides["output_dir"] = output_dir.strip()
    for field, raw, cast, label in (
        ("steps", steps, int, "步数"),
        ("rank", rank, int, "rank"),
        ("alpha", alpha, int, "alpha"),
        ("batch_size", batch_size, int, "batch"),
        ("lr", lr, float, "学习率"),
    ):
        value, error = _parse_number(raw, label, cast)
        if error:
            errors.append(error)
        elif value is not None:
            overrides[field] = value
    if errors:
        return None, errors
    try:
        plan = plan_for_tier(target, vram_tier, **overrides)
    except ValueError as exc:
        return None, [str(exc)]
    return plan, []


def acceptance_markdown(target: TargetModel, plan: Any) -> str:
    """验收清单的 LF 版本。

    渲染在 ``loraforge.configs.acceptance``（阿配的写集），界面只保证换行口径：
    纯文本 → 统一 LF + 结尾换行，这样界面预览与 zip 里的 ``验收.md`` 是同一份字节。
    """
    text = render_acceptance_markdown(target, plan)
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"


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
            "saved": load_state(),
            "offload_max_gb": OFFLOAD_MAX_GB,
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


@app.post("/review")
def review(
    dataset_dir: str = Form(...),
    model: str = Form(...),
    kind: str = Form(...),
    trigger: str = Form(""),
) -> JSONResponse:
    """逐图校对：打标词 + 库给出的 severity / message，界面只负责上色与编辑。"""
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
        items = review_items(raw, target, CaptionContext(target=target, kind=dataset_kind, trigger=trigger.strip()))
    except ExportError as exc:
        return _fail("参数", [str(exc)])
    if not items:
        return _fail("参数", [f"目录里没有可校对的图（只认 .jpg/.jpeg/.png）：{Path(raw).expanduser()}"])
    return JSONResponse(
        {
            "ok": True,
            "root": str(Path(raw).expanduser()),
            "count": len(items),
            "summary": severity_counts(items),
            "items": items,
        }
    )


@app.post("/caption/save")
def save_caption(
    dataset_dir: str = Form(...),
    image: str = Form(...),
    caption: str = Form(""),
    model: str = Form("sdxl"),
    kind: str = Form("character"),
    trigger: str = Form(""),
) -> JSONResponse:
    """保存一张图的打标词：bytes 写、LF 换行、临时文件 + os.replace。"""
    raw = dataset_dir.strip()
    name = image.strip()
    if not raw or not name:
        return _fail("参数", ["图集目录与图名都不能为空"])
    try:
        scan_result = scan_images(Path(raw).expanduser())
    except ExportError as exc:
        return _fail("参数", [str(exc)])
    entry = next((item for item in scan_result.images if item.name == name), None)
    if entry is None:
        return _fail("参数", [f"目录里没有这张图：{name}"])

    try:
        target = TargetModel(model)
        dataset_kind = DatasetKind(kind)
    except ValueError as exc:
        return _fail("参数", [str(exc)])

    body = caption.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"
    txt = entry.path.with_suffix(".txt")
    tmp = txt.with_name(txt.name + ".tmp")
    tmp.write_bytes(body.encode("utf-8"))
    os.replace(tmp, txt)
    pairs = [(item.path, _caption_text(item.path)) for item in scan_result.images]
    ctx = CaptionContext(target=target, kind=dataset_kind, trigger=trigger.strip())
    check = next(check for check in check_captions(pairs, ctx) if check.image == entry.path)
    return JSONResponse(
        {
            "ok": True,
            "image": name,
            "txt": str(txt),
            "bytes": len(body.encode("utf-8")),
            "check": {
                "code": check.code,
                "severity": check.severity,
                "severity_label": SEVERITY_LABEL.get(check.severity, check.severity),
                "message": check.message,
            },
        }
    )


@app.get("/asset")
def asset(dataset_dir: str = Query(...), name: str = Query(...)) -> FileResponse:
    """校对页缩略图：只给图集目录顶层、只给三种图片后缀。"""
    root = Path(dataset_dir.strip()).expanduser().resolve()
    if not root.is_dir():
        return _fail("参数", [f"图集目录不存在：{root}"])
    # 只认顶层文件名：带任何分隔符或 .. 的直接拒，不让它靠 resolve 碰运气
    if Path(name).name != name or name in {".", ".."}:
        return _fail("参数", [f"拒绝读取：{name}"])
    candidate = (root / name).resolve()
    if (
        candidate.parent != root
        or candidate.suffix.lower() not in {".jpg", ".jpeg", ".png"}
        or not candidate.is_file()
    ):
        return _fail("参数", [f"拒绝读取：{name}"])
    return FileResponse(candidate)


@app.post("/config/preview")
def preview_config(
    model: str = Form(...),
    vram_tier: str = Form("24"),
    dataset_dir: str = Form(""),
    trigger: str = Form(""),
    output_dir: str = Form(""),
    steps: str = Form(""),
    rank: str = Form(""),
    alpha: str = Form(""),
    lr: str = Form(""),
    batch_size: str = Form(""),
) -> JSONResponse:
    """参数 → plan → config → validate_config_detailed；只有 error 才禁导出。"""
    try:
        target = TargetModel(model)
    except ValueError:
        return _fail("参数", [f"未知目标模型：{model!r}"])
    plan, errors = build_plan_from_form(
        target,
        vram_tier,
        dataset_dir=dataset_dir,
        trigger=trigger,
        output_dir=output_dir,
        steps=steps,
        rank=rank,
        alpha=alpha,
        lr=lr,
        batch_size=batch_size,
    )
    if errors:
        return _fail("参数", errors)

    config = build_config(target, plan)
    problems = validate_config_detailed(config, target)
    payload = {
        "problems": [
            {"code": problem.code, "severity": problem.severity, "message": problem.message}
            for problem in problems
        ],
    }
    payload["problem_groups"] = [
        {
            "severity": severity,
            "label": SEVERITY_LABEL[severity],
            "items": [item for item in payload["problems"] if item["severity"] == severity],
        }
        for severity in ("error", "warn", "info")
    ]
    return JSONResponse(
        {
            "ok": not any(problem["severity"] == "error" for problem in payload["problems"]),
            "can_export": not any(problem["severity"] == "error" for problem in payload["problems"]),
            "plan": {
                "model_path": plan.model_path,
                "dataset_dir": plan.dataset_dir,
                "output_dir": plan.output_dir,
                "trigger": plan.trigger,
                "steps": plan.steps,
                "rank": plan.rank,
                "alpha": plan.alpha,
                "lr": plan.lr,
                "batch_size": plan.batch_size,
                "resolutions": list(plan.resolutions),
                "vram_gb": plan.vram_gb,
                "vram_tier": plan.vram_tier if hasattr(plan, "vram_tier") else vram_tier,
            },
            "problems": payload["problems"],
            "problem_groups": payload["problem_groups"],
            "yaml": yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        }
    )


@app.post("/acceptance")
def acceptance(
    model: str = Form(...),
    vram_tier: str = Form("24"),
    dataset_dir: str = Form(""),
    trigger: str = Form(""),
    output_dir: str = Form(""),
    steps: str = Form(""),
    rank: str = Form(""),
    alpha: str = Form(""),
    lr: str = Form(""),
    batch_size: str = Form(""),
) -> JSONResponse:
    """按当前参数渲染验收清单（Markdown）。导出时会以 ``验收.md`` 进 zip。"""
    try:
        target = TargetModel(model)
    except ValueError:
        return _fail("参数", [f"未知目标模型：{model!r}"])
    plan, errors = build_plan_from_form(
        target,
        vram_tier,
        dataset_dir=dataset_dir,
        trigger=trigger,
        output_dir=output_dir,
        steps=steps,
        rank=rank,
        alpha=alpha,
        lr=lr,
        batch_size=batch_size,
    )
    if errors:
        return _fail("参数", errors)
    markdown = acceptance_markdown(target, plan)
    blob = markdown.encode("utf-8")
    return JSONResponse(
        {
            "ok": True,
            "model": target.value,
            "trigger": plan.trigger,
            "bytes": len(blob),
            "lines": markdown.count("\n"),
            "markdown": markdown,
            "html": render_markdown(markdown),
        }
    )


@app.get("/state")
def get_state() -> JSONResponse:
    return JSONResponse(load_state())


@app.post("/state/save")
async def save_state(request: Request) -> JSONResponse:
    """存表单值。只收 STATE_FIELDS 里的字符串，别的一律丢掉并说明。"""
    try:
        payload = await request.json()
    except ValueError:
        return _fail("参数", ["请求体不是 JSON"])
    values = payload.get("values") if isinstance(payload, dict) else None
    if not isinstance(values, dict):
        return _fail("参数", ["values 必须是对象"])
    kept = {
        key: str(value)
        for key, value in values.items()
        if key in STATE_FIELDS and isinstance(value, (str, int, float)) and not isinstance(value, bool)
    }
    dropped = sorted(set(values) - set(kept))
    saved = _write_state(kept)
    return JSONResponse(
        {
            "ok": True,
            "schema": saved["schema"],
            "saved_at": saved["saved_at"],
            "path": str(state_path()),
            "saved": sorted(kept),
            "dropped": dropped,
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
    problems = validate_config_detailed(config, target)
    hard_errors = [problem.message for problem in problems if problem.severity == "error"]
    if hard_errors:
        # 危险配置：不生成 yaml、不打包，直接把问题摆出来。
        return _fail("校验", hard_errors)
    warnings = [
        {"code": problem.code, "severity": problem.severity, "message": problem.message}
        for problem in problems
        if problem.severity != "error"
    ]

    out = Path(out_zip.strip()).expanduser()
    if out.suffix.lower() != ".zip":
        return _fail("参数", [f"导出路径要以 .zip 结尾：{out}"])

    out.parent.mkdir(parents=True, exist_ok=True)
    config_path = out.parent / "train.yaml"
    config_path.write_bytes(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False).encode("utf-8")
    )

    acceptance_text = acceptance_markdown(target, plan)
    with_images = include_images.strip().lower() not in {"", "0", "false", "off"}
    try:
        report = package(
            dataset,
            config_path,
            out,
            include_images=with_images,
            extra_files={"验收.md": acceptance_text.encode("utf-8")},
        )
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
            "acceptance": report.extra_files,
            "acceptance_bytes": len(acceptance_text.encode("utf-8")),
            "with_images": with_images,
            "warnings": warnings,
        }
    )


__all__ = ["app", "backend_rows", "model_cards", "vram_rows"]
