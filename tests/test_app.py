"""界面冒烟测试：首页能开、三款模型与后端清单都渲染、导出接口能跑通。

不联网、不碰 GPU、不读真实图集 —— 目录与图片都在 ``.scratch`` 里临时生成。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as ui
from loraforge.caption.base import BackendCaps
from loraforge.caption.pipeline import BackendInfo
from loraforge.models import PROFILES
from loraforge.testing import FakeBackend

CLIENT = TestClient(ui.app)

CAPS = BackendCaps(needs_gpu=False, supports_tags=True, supports_prose=True, languages=("en",))


def test_index_lists_all_three_models() -> None:
    response = CLIENT.get("/")
    assert response.status_code == 200
    for profile in PROFILES.values():
        assert profile.label in response.text
        assert profile.train_base in response.text


def test_index_renders_defaults_from_configs() -> None:
    """参数默认值必须来自 default_plan，页面里的步数不是界面写死的。"""
    from loraforge.configs import default_plan

    response = CLIENT.get("/")
    steps = str(default_plan("sdxl").steps)
    assert steps in response.text


def test_index_lists_unavailable_backend_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    infos = [
        BackendInfo(key="fake", label="Fake（测试用）", caps=CAPS, available=True, reason="ok"),
        BackendInfo(
            key="onnx-tagger",
            label="ONNX Tagger",
            caps=BackendCaps(
                needs_gpu=False, supports_tags=True, supports_prose=False, languages=("en",)
            ),
            available=False,
            reason="missing_dependency:onnxruntime: 本机没装 onnxruntime，无法本地打标",
        ),
    ]
    monkeypatch.setattr(ui, "_backends", lambda: infos)

    response = CLIENT.get("/")

    assert response.status_code == 200
    assert "onnx-tagger" in response.text          # 不可用的也列出来
    assert "不可用" in response.text
    assert "本机没装 onnxruntime" in response.text  # 原因写在行上，不是只打点


def test_healthz_reports_models_and_backends() -> None:
    response = CLIENT.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert len(payload["models"]) == 3                                  # 恰好三款，不多不少
    assert set(payload["models"]) == {target.value for target in PROFILES}
    # 后端清单跟着 list_backends() 走：这里只断言形状，不钉死具体有哪些后端
    assert isinstance(payload["backends"], list)
    assert payload["backends"] == [info.key for info in ui._backends()]


def test_export_endpoint_writes_zip_and_reports_missing(scratch: Path) -> None:
    dataset = scratch / "my_dataset"
    dataset.mkdir()
    for name in ("frame_01.png", "frame_02.png"):
        (dataset / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    (dataset / "frame_01.txt").write_text(
        "mytoken, a photo of a person, standing\n", encoding="utf-8"
    )
    out_zip = scratch / "out.zip"

    response = CLIENT.post(
        "/export",
        data={
            "dataset_dir": str(dataset),
            "out_zip": str(out_zip),
            "model": "sdxl",
            "trigger": "mytoken",
            "vram_tier": "24",
            "include_images": "1",
            "allow_missing": "1",                       # 显式放行才会出包
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is False                      # 缺一张标，不算干净
    assert payload["allow_missing"] is True
    assert payload["missing_caption"] == ["frame_02.png"]
    assert payload["images"] == 2 and payload["captions"] == 1
    assert Path(payload["config"]).is_file()
    assert out_zip.is_file()
    with zipfile.ZipFile(out_zip) as archive:
        assert "my_dataset/train.yaml" in archive.namelist()


def test_export_rejects_missing_trigger(scratch: Path) -> None:
    dataset = scratch / "ds"
    dataset.mkdir()
    (dataset / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    response = CLIENT.post(
        "/export",
        data={
            "dataset_dir": str(dataset),
            "out_zip": str(scratch / "out.zip"),
            "model": "sdxl",
            "trigger": "   ",
            "vram_tier": "24",
        },
    )

    assert response.status_code == 400
    assert "触发词不能为空" in response.json()["errors"][0]


def test_export_rejects_unknown_model_and_tier(scratch: Path) -> None:
    dataset = scratch / "ds"
    dataset.mkdir()
    (dataset / "a.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    base = {
        "dataset_dir": str(dataset),
        "out_zip": str(scratch / "out.zip"),
        "trigger": "mytoken",
    }

    bad_model = CLIENT.post("/export", data={**base, "model": "flux", "vram_tier": "24"})
    assert bad_model.status_code == 400
    assert "未知目标模型" in bad_model.json()["errors"][0]

    bad_tier = CLIENT.post("/export", data={**base, "model": "sdxl", "vram_tier": "999"})
    assert bad_tier.status_code == 400
    assert "显存档位" in bad_tier.json()["errors"][0]


def test_export_bad_params_are_4xx_never_500(scratch: Path) -> None:
    """非法参数一律 4xx 且带中文原因，绝不允许 500 把栈甩给用户。"""
    dataset = scratch / "ds"
    dataset.mkdir()
    (dataset / "a.png").write_bytes(b"\\x89PNG\\r\\n\\x1a\\n")
    good = {
        "dataset_dir": str(dataset),
        "out_zip": str(scratch / "out.zip"),
        "model": "sdxl",
        "trigger": "mytoken",
        "vram_tier": "24",
    }
    cases = {
        "目录不存在": {**good, "dataset_dir": str(scratch / "no_such_dir")},
        "输出不是 zip": {**good, "out_zip": str(scratch / "out.tar")},
        "未知模型": {**good, "model": "flux"},
        "未知档位": {**good, "vram_tier": "999"},
        "触发词空白": {**good, "trigger": "  "},
    }
    for name, payload in cases.items():
        response = CLIENT.post("/export", data=payload)
        assert 400 <= response.status_code < 500, f"{name} 返回了 {response.status_code}"
        assert response.json()["errors"], f"{name} 没给原因"
    assert not (scratch / "out.zip").exists()           # 一条都不该写出成品


# ---------------------------------------------------------------------------
# PR-B1：图集清点 + 打标跑批
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_backend_cache():
    """每个用例前后清掉后端实例缓存，避免用例之间互相复用假后端。"""
    ui._forget_backends()
    yield
    ui._forget_backends()


def _dataset_with_images(scratch: Path, names: tuple[str, ...], with_caption=()) -> Path:
    dataset = scratch / "ds"
    dataset.mkdir()
    for name in names:
        (dataset / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    for name in with_caption:
        (dataset / name).write_text("mytoken, a photo of a person\n", encoding="utf-8")
    return dataset


def test_scan_lists_images_and_reports_skipped(scratch: Path) -> None:
    dataset = _dataset_with_images(
        scratch, ("a.png", "b.JPG", "c.jpeg", "d.webp"), with_caption=("a.txt",)
    )
    (dataset / "notes.txt").write_text("x\n", encoding="utf-8")
    (dataset / "sub").mkdir()

    response = CLIENT.post("/scan", data={"dataset_dir": str(dataset)})

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["count"] == 3                                  # d.webp 不算训练图
    names = [item["name"] for item in payload["images"]]
    assert names == ["a.png", "b.JPG", "c.jpeg"]                   # 后缀大小写不敏感
    assert [item["has_caption"] for item in payload["images"]] == [True, False, False]
    assert payload["with_caption"] == 1
    skipped = set(payload["skipped"]["names"])
    assert {"d.webp", "notes.txt", "sub"} <= skipped                # 跳过的必须点名
    assert payload["skipped"]["count"] == len(skipped)


def test_scan_rejects_missing_dir(scratch: Path) -> None:
    response = CLIENT.post("/scan", data={"dataset_dir": str(scratch / "nope")})
    assert response.status_code == 400
    assert "图集目录不存在" in response.json()["errors"][0]


def test_caption_writes_txt_and_reports_counts(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = _dataset_with_images(scratch, ("frame_01.png", "frame_02.png", "frame_03.png"))
    monkeypatch.setattr(ui, "get_backend", lambda key: FakeBackend())

    response = CLIENT.post(
        "/caption",
        data={
            "dataset_dir": str(dataset),
            "model": "sdxl",
            "kind": "character",
            "trigger": "mytoken",
            "backend": "wd_tagger",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert (payload["written"], payload["failed"], payload["skipped"]) == (3, 0, 0)
    assert payload["ok"] is True
    assert all(item["status"] == "written" for item in payload["items"])
    for name in ("frame_01.txt", "frame_02.txt", "frame_03.txt"):
        text = (dataset / name).read_text(encoding="utf-8")
        assert text.strip().startswith("mytoken")        # 触发词被 styler 注进第一位


def test_caption_reports_failed_item_with_backend_reason(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset_with_images(scratch, ("frame_01.png", "frame_02.png"))
    monkeypatch.setattr(ui, "get_backend", lambda key: FakeBackend(fail_on={"frame_02"}))

    response = CLIENT.post(
        "/caption",
        data={"dataset_dir": str(dataset), "model": "moss_image", "kind": "style", "backend": "wd_tagger"},
    )

    payload = response.json()
    assert response.status_code == 200
    assert (payload["written"], payload["failed"]) == (1, 1)
    assert payload["ok"] is False
    failed = payload["failed_items"]
    assert [item["image"] for item in failed] == ["frame_02.png"]
    assert failed[0]["error"] == "fake backend 故意失败：frame_02"  # reason 原文，不摘要
    assert failed[0]["retryable"] is True
    status_by_name = {item["image"]: item["status"] for item in payload["items"]}
    assert status_by_name == {"frame_01.png": "written", "frame_02.png": "failed"}


def test_caption_retry_reruns_only_selected_images(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset_with_images(scratch, ("frame_01.png", "frame_02.png", "frame_03.png"))
    monkeypatch.setattr(ui, "get_backend", lambda key: FakeBackend(fail_on={"frame_02"}))

    first = CLIENT.post(
        "/caption",
        data={"dataset_dir": str(dataset), "model": "sdxl", "kind": "character", "backend": "wd_tagger"},
    ).json()
    assert first["failed"] == 1

    monkeypatch.setattr(ui, "get_backend", lambda key: FakeBackend())
    ui._forget_backends()                                          # 等价于用户换了后端/权重
    retry = CLIENT.post(
        "/caption",
        data={
            "dataset_dir": str(dataset),
            "model": "sdxl",
            "kind": "character",
            "backend": "wd_tagger",
            "images": "frame_02.png",
        },
    ).json()

    assert retry["total"] == 1                                     # 只跑失败那张
    assert retry["written"] == 1 and retry["failed"] == 0
    assert (dataset / "frame_02.txt").is_file()


def test_caption_reuses_backend_instance_across_requests(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset_with_images(scratch, ("frame_01.png",))
    calls: list[str] = []

    def counting(key: str):
        calls.append(key)
        return FakeBackend()

    monkeypatch.setattr(ui, "get_backend", counting)

    for _ in range(3):
        CLIENT.post(
            "/caption",
            data={"dataset_dir": str(dataset), "model": "sdxl", "kind": "character", "backend": "wd_tagger"},
        )

    assert calls == ["wd_tagger"]                                   # 只建一次后端


def test_caption_rejects_unavailable_backend_with_reason(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset_with_images(scratch, ("frame_01.png",))

    def broken(key: str):
        raise RuntimeError("缺少 onnxruntime；无法运行 WD tagger")

    monkeypatch.setattr(ui, "get_backend", broken)

    response = CLIENT.post(
        "/caption",
        data={"dataset_dir": str(dataset), "model": "sdxl", "kind": "character", "backend": "wd_tagger"},
    )

    assert response.status_code == 400
    assert response.json()["stage"] == "后端"
    assert "缺少 onnxruntime" in response.json()["errors"][0]


def test_index_has_caption_section() -> None:
    page = CLIENT.get("/").text
    assert "图集与打标" in page
    assert 'id="scan-form"' in page
    assert 'id="caption-form"' in page
    assert "已跳过" in page


def test_caption_skips_existing_caption_when_overwrite_off(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = _dataset_with_images(scratch, ("frame_01.png", "frame_02.png"), with_caption=("frame_01.txt",))
    monkeypatch.setattr(ui, "get_backend", lambda key: FakeBackend())

    payload = CLIENT.post(
        "/caption",
        data={
            "dataset_dir": str(dataset),
            "model": "sdxl",
            "kind": "character",
            "trigger": "mytoken",
            "backend": "wd_tagger",
            "overwrite": "0",
        },
    ).json()

    assert (payload["written"], payload["skipped"], payload["failed"]) == (1, 1, 0)
    status = {item["image"]: item["status"] for item in payload["items"]}
    assert status == {"frame_01.png": "skipped", "frame_02.png": "written"}
    assert (dataset / "frame_01.txt").read_text(encoding="utf-8").strip() == "mytoken, a photo of a person"


# ---------------------------------------------------------------------------
# B2：逐张校对、参数预览、配置存盘
# ---------------------------------------------------------------------------


def _caption(dataset: Path, name: str, text: str) -> None:
    (dataset / f"{Path(name).stem}.txt").write_bytes(text.encode("utf-8"))


def _review(dataset: Path, **extra) -> dict:
    data = {"dataset_dir": str(dataset), "model": "sdxl", "kind": "character", "trigger": "mytoken"}
    data.update(extra)
    response = CLIENT.post("/review", data=data)
    assert response.status_code == 200, response.text
    return response.json()


def test_review_passes_library_severity_and_message_through(scratch: Path) -> None:
    """上色依据是 severity + message 原文，和 check_captions 逐字一致。"""
    from loraforge.caption.base import CaptionContext
    from loraforge.caption.checks import check_captions
    from loraforge.models import DatasetKind, TargetModel

    dataset = _dataset_with_images(scratch, ("a.png", "b.png", "c.png"))
    _caption(dataset, "b.png", "mytoken, cat")            # 标签过少 -> warn
    _caption(dataset, "c.png", "mytoken, cat")            # 与 b 完全一致 -> warn
    payload = _review(dataset)

    by_name = {item["image"]: item for item in payload["items"]}
    assert set(by_name) == {"a.png", "b.png", "c.png"}
    # 缺 .txt 就是空打标词，红，不是「跳过」
    assert (by_name["a.png"]["code"], by_name["a.png"]["severity"]) == ("empty", "error")
    assert by_name["a.png"]["message"]                      # 必须有原因文字

    names = sorted(by_name)
    pairs = [(dataset / name, by_name[name]["caption"]) for name in names]
    ctx = CaptionContext(target=TargetModel.SDXL, kind=DatasetKind.CHARACTER, trigger="mytoken")
    expected = {check.image.name: check for check in check_captions(pairs, ctx)}
    for name in names:
        assert by_name[name]["severity"] == expected[name].severity
        assert by_name[name]["message"] == expected[name].message
        assert by_name[name]["severity_label"] in {"红", "黄", "正常"}

    assert payload["summary"] == {"error": 1, "warn": 2, "info": 0}


def test_review_rejects_missing_dataset(scratch: Path) -> None:
    response = CLIENT.post(
        "/review",
        data={"dataset_dir": str(scratch / "nope"), "model": "sdxl", "kind": "character", "trigger": "mytoken"},
    )
    assert response.status_code == 400
    assert response.json()["errors"]


def test_save_caption_writes_lf_bytes_and_rechecks(scratch: Path) -> None:
    dataset = _dataset_with_images(scratch, ("b.png",))
    _caption(dataset, "b.png", "old")

    response = CLIENT.post(
        "/caption/save",
        data={
            "dataset_dir": str(dataset),
            "image": "b.png",
            "caption": "mytoken, cat, sitting, indoors\n\n",
            "model": "sdxl",
            "kind": "character",
            "trigger": "mytoken",
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    raw = (dataset / "b.txt").read_bytes()
    assert raw == b"mytoken, cat, sitting, indoors\n"      # 归一成单个 LF，不写 CRLF
    assert b"\r" not in raw
    assert body["bytes"] == len(raw)
    assert body["check"]["severity"] == "info"               # 保存后按库重新判级
    assert not (dataset / "b.txt.tmp").exists()


def test_save_caption_rejects_unknown_image(scratch: Path) -> None:
    dataset = _dataset_with_images(scratch, ("b.png",))
    response = CLIENT.post(
        "/caption/save",
        data={"dataset_dir": str(dataset), "image": "ghost.png", "caption": "x", "model": "sdxl", "kind": "character"},
    )
    assert response.status_code == 400
    assert not (dataset / "ghost.txt").exists()


def test_asset_only_serves_top_level_images(scratch: Path) -> None:
    dataset = _dataset_with_images(scratch, ("b.png",))
    (dataset / "notes.txt").write_text("x", encoding="utf-8")

    ok = CLIENT.get("/asset", params={"dataset_dir": str(dataset), "name": "b.png"})
    assert ok.status_code == 200

    for bad in ("../b.png", "notes.txt", "sub/../b.png"):
        blocked = CLIENT.get("/asset", params={"dataset_dir": str(dataset), "name": bad})
        assert blocked.status_code == 400, bad


def test_config_preview_reads_defaults_from_configs(scratch: Path) -> None:
    from loraforge.configs import VRAM_TIERS, default_plan

    dataset = _dataset_with_images(scratch, ("b.png",), with_caption=("b.txt",))
    response = CLIENT.post(
        "/config/preview",
        data={"model": "sdxl", "vram_tier": "12", "dataset_dir": str(dataset), "trigger": "mytoken"},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    plan = default_plan("sdxl")
    assert data["plan"]["steps"] == plan.steps                # 空字段 = 库默认
    assert data["plan"]["rank"] == plan.rank
    assert data["plan"]["batch_size"] == VRAM_TIERS["12"].batch_size
    assert data["can_export"] is True                          # 只有 warn，不禁导出
    assert any(item["severity"] == "warn" for item in data["problems"])
    assert "config:" in data["yaml"]


def test_config_preview_blocks_export_only_on_error(scratch: Path) -> None:
    dataset = _dataset_with_images(scratch, ("b.png",), with_caption=("b.txt",))

    bad = CLIENT.post(
        "/config/preview",
        data={"model": "sdxl", "vram_tier": "24", "dataset_dir": str(dataset), "trigger": "mytoken", "steps": "0"},
    ).json()
    assert bad["can_export"] is False
    assert any(item["severity"] == "error" for item in bad["problems"])
    assert any(item["code"] == "train.steps" for item in bad["problems"])

    good = CLIENT.post(
        "/config/preview",
        data={"model": "sdxl", "vram_tier": "24", "dataset_dir": str(dataset), "trigger": "mytoken", "steps": "2500"},
    ).json()
    assert good["can_export"] is True


def test_config_preview_rejects_non_numeric_steps(scratch: Path) -> None:
    dataset = _dataset_with_images(scratch, ("b.png",))
    response = CLIENT.post(
        "/config/preview",
        data={"model": "sdxl", "vram_tier": "24", "dataset_dir": str(dataset), "trigger": "mytoken", "steps": "abc"},
    )
    assert response.status_code == 400
    assert "abc" in response.json()["errors"][0]


def test_page_has_review_preview_refresh_and_offload_hint() -> None:
    page = CLIENT.get("/").text
    assert 'id="review-form"' in page
    assert 'id="preview-btn"' in page
    assert 'id="cap_refresh"' in page
    assert "保存这张" in page
    # 8 / 12 GB 两档各一句人话；16 GB 也开量化但按主管要求只提示这两档
    assert page.count(ui.OFFLOAD_HINT) == 2
    assert ui.OFFLOAD_MAX_GB == 12


def test_export_reports_warnings_without_blocking(scratch: Path) -> None:
    dataset = _dataset_with_images(scratch, ("b.png",), with_caption=("b.txt",))
    out_zip = scratch / "out.zip"

    response = CLIENT.post(
        "/export",
        data={
            "dataset_dir": str(dataset),
            "out_zip": str(out_zip),
            "model": "sdxl",
            "trigger": "mytoken",
            "vram_tier": "24",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    assert any(item["severity"] == "warn" for item in payload["warnings"])
    assert any(item["code"] == "datasets.resolution-parallel" for item in payload["warnings"])


def test_state_roundtrip_writes_schema_and_prefills_page(
    scratch: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_file = scratch / "ui_state.json"
    monkeypatch.setenv(ui.STATE_ENV, str(state_file))

    saved = CLIENT.post(
        "/state/save",
        json={"values": {"model": "krea2", "trigger": "xyzzy", "vram_tier": "12", "bogus": "1"}},
    )
    assert saved.status_code == 200, saved.text
    body = saved.json()
    assert body["schema"] == 1
    assert body["dropped"] == ["bogus"]
    on_disk = state_file.read_bytes()
    assert b'"schema": 1' in on_disk
    assert b"xyzzy" in on_disk

    loaded = CLIENT.get("/state").json()
    assert loaded["ok"] is True
    assert loaded["values"]["trigger"] == "xyzzy"

    page = CLIENT.get("/").text
    assert 'value="xyzzy"' in page
    assert '<option value="krea2" selected>' in page


def test_state_rejects_unknown_schema(scratch: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    state_file = scratch / "ui_state.json"
    state_file.write_bytes(b'{"schema": 0, "values": {"trigger": "oldvalue"}}')
    monkeypatch.setenv(ui.STATE_ENV, str(state_file))

    loaded = CLIENT.get("/state").json()
    assert loaded["ok"] is False
    assert "schema" in loaded["errors"][0]
    assert loaded["values"] == {}
    assert 'value="oldvalue"' not in CLIENT.get("/").text


def test_acceptance_endpoint_renders_markdown(scratch: Path) -> None:
    """验收清单由 loraforge.configs.acceptance 渲染，界面只透传 + 保证 LF。"""
    response = CLIENT.post(
        "/acceptance",
        data={
            "model": "sdxl",
            "vram_tier": "24",
            "trigger": "mytoken",
            "dataset_dir": str(scratch),
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    markdown = payload["markdown"]
    assert markdown.startswith("# SDXL LoRA 验收清单")
    assert "mytoken" in markdown
    assert "\r" not in markdown
    assert markdown.endswith("\n")
    assert payload["bytes"] == len(markdown.encode("utf-8"))
    assert payload["lines"] == markdown.count("\n")


def test_acceptance_endpoint_rejects_unknown_model() -> None:
    response = CLIENT.post("/acceptance", data={"model": "nope"})
    assert 400 <= response.status_code < 500
    assert response.json()["ok"] is False
    assert response.json()["errors"]


def test_export_zip_contains_acceptance_markdown(scratch: Path) -> None:
    """导出 zip 里带 验收.md，内容与 /acceptance 预览逐字节一致、且是 LF。"""
    dataset = scratch / "my_dataset"
    dataset.mkdir()
    for name in ("frame_01.png", "frame_02.png"):
        (dataset / name).write_bytes(b"\x89PNG\r\n\x1a\n")
        (dataset / name).with_suffix(".txt").write_text(
            "mytoken, a photo of a person, standing\n", encoding="utf-8"
        )
    out_zip = scratch / "out.zip"

    preview = CLIENT.post(
        "/acceptance",
        data={
            "model": "moss_image",
            "vram_tier": "16",
            "trigger": "mytoken",
            "dataset_dir": str(dataset),
        },
    )
    assert preview.status_code == 200, preview.text
    markdown = preview.json()["markdown"]

    response = CLIENT.post(
        "/export",
        data={
            "dataset_dir": str(dataset),
            "out_zip": str(out_zip),
            "model": "moss_image",
            "trigger": "mytoken",
            "vram_tier": "16",
            "include_images": "0",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    assert payload["acceptance"] == ["验收.md"]
    assert payload["acceptance_bytes"] == len(markdown.encode("utf-8"))
    with zipfile.ZipFile(out_zip) as archive:
        names = archive.namelist()
        assert "my_dataset/验收.md" in names
        blob = archive.read("my_dataset/验收.md")
    assert blob == markdown.encode("utf-8")
    assert b"\r" not in blob


def test_acceptance_html_renders_and_escapes_trigger(scratch: Path) -> None:
    """验收清单同时给 Markdown 与渲染后的 HTML；触发词里的标签不能变成真标签。"""
    response = CLIENT.post(
        "/acceptance",
        data={
            "model": "sdxl",
            "vram_tier": "24",
            "trigger": "<script>alert(1)</script>",
            "dataset_dir": str(scratch),
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "<h1>" in payload["html"] and "<table>" in payload["html"]
    assert "<script>" not in payload["html"]
    assert "&lt;script&gt;" in payload["html"]


def test_export_train_yaml_is_byte_identical_to_render_train_config(scratch: Path) -> None:
    """接线的全部意义：zip 里的 train.yaml 就是 render_train_config 的返回字节。"""
    from loraforge.configs import plan_for_tier, render_train_config

    dataset = _dataset_with_images(scratch, ("a.png",), with_caption=("a.txt",))
    out_zip = scratch / "out.zip"
    form = {
        "dataset_dir": str(dataset),
        "out_zip": str(out_zip),
        "model": "sdxl",
        "trigger": "mytoken",
        "vram_tier": "24",
        "include_images": "1",
    }

    response = CLIENT.post("/export", data=form)
    assert response.status_code == 200, response.text

    plan = plan_for_tier("sdxl", "24", dataset_dir=str(dataset), trigger="mytoken")
    expected = render_train_config("sdxl", plan)
    assert b"\r" not in expected and expected.endswith(b"\n")
    assert expected.startswith(b"#")          # header 三行注释在包里那份里

    with zipfile.ZipFile(out_zip) as archive:
        assert archive.read(f"{dataset.name}/train.yaml") == expected

    # 预览与导出一致：同一个函数、同一组参数，字节必须相同。
    preview = CLIENT.post(
        "/config/preview",
        data={"model": "sdxl", "vram_tier": "24", "dataset_dir": str(dataset), "trigger": "mytoken"},
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["yaml"].encode("utf-8") == expected


def test_export_blocks_missing_captions_unless_allowed(scratch: Path) -> None:
    """缺打标词默认 400：云上开训才发现一半图没标，是最贵的失败。"""
    dataset = scratch / "ds"
    dataset.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        (dataset / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    (dataset / "a.txt").write_bytes(b"mytoken, a photo of a person, standing\n")
    out_zip = scratch / "out.zip"
    form = {
        "dataset_dir": str(dataset),
        "out_zip": str(out_zip),
        "model": "sdxl",
        "trigger": "mytoken",
        "vram_tier": "24",
    }

    blocked = CLIENT.post("/export", data=form)
    assert blocked.status_code == 400, blocked.text
    body = blocked.json()
    assert body["stage"] == "缺打标词"
    text = " ".join(body["errors"])
    assert "2 张图没有打标词" in text
    assert "b.png" in text and "c.png" in text
    assert "allow_missing=1" in text
    assert not out_zip.exists()                        # 拦下来就没落盘

    allowed = CLIENT.post("/export", data={**form, "allow_missing": "1"})
    assert allowed.status_code == 200, allowed.text
    payload = allowed.json()
    assert payload["ok"] is False
    assert payload["missing_caption"] == ["b.png", "c.png"]
    assert payload["allow_missing"] is True
    assert out_zip.is_file()

    (dataset / "b.txt").write_bytes(b"mytoken, a photo of a person, sitting\n")
    (dataset / "c.txt").write_bytes(b"mytoken, a photo of a person, walking\n")
    clean = CLIENT.post("/export", data={**form, "out_zip": str(scratch / "clean.zip")})
    assert clean.status_code == 200, clean.text
    assert clean.json()["ok"] is True
    assert clean.json()["missing_caption"] == []


def test_export_honours_custom_plan_fields(scratch: Path) -> None:
    """参数面板改的 steps/rank/alpha/lr/batch 必须进 zip 的 yaml，且与预览逐字节一致。"""
    from loraforge.configs import plan_for_tier, render_train_config

    dataset = _dataset_with_images(scratch, ("a.png",), with_caption=("a.txt",))
    out_zip = scratch / "out.zip"
    custom = {"steps": "2000", "rank": "16", "alpha": "16", "lr": "5e-5", "batch_size": "2"}
    form = {
        "dataset_dir": str(dataset),
        "out_zip": str(out_zip),
        "model": "sdxl",
        "trigger": "mytoken",
        "vram_tier": "24",
        **custom,
    }

    response = CLIENT.post("/export", data=form)
    assert response.status_code == 200, response.text

    plan = plan_for_tier(
        "sdxl",
        "24",
        dataset_dir=str(dataset),
        trigger="mytoken",
        steps=2000,
        rank=16,
        alpha=16,
        lr=5e-5,
        batch_size=2,
    )
    expected = render_train_config("sdxl", plan)
    assert b"steps: 2000" in expected
    with zipfile.ZipFile(out_zip) as archive:
        blob = archive.read(f"{dataset.name}/train.yaml")
    assert blob == expected

    preview = CLIENT.post(
        "/config/preview",
        data={"model": "sdxl", "vram_tier": "24", "dataset_dir": str(dataset),
              "trigger": "mytoken", **custom},
    )
    assert preview.status_code == 200, preview.text
    assert preview.json()["plan"]["steps"] == 2000
    assert preview.json()["plan"]["rank"] == 16
    assert preview.json()["yaml"].encode("utf-8") == expected


def test_review_flags_krea_identity_removed(scratch: Path) -> None:
    """Krea 标词里写了身份成分 → 页面要真出现 identity_removed 的黄字。"""
    dataset = _dataset_with_images(scratch, ("a.png",))
    (dataset / "a.txt").write_bytes(
        b"mytoken, a 25 year old woman with long black hair and blue eyes, "
        b"wearing a red dress, smiling in a cafe\n"
    )

    response = CLIENT.post(
        "/review",
        data={"dataset_dir": str(dataset), "model": "krea2", "kind": "character", "trigger": "mytoken"},
    )

    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["code"] == "identity_removed"
    assert item["severity"] == "warn"
    assert item["severity_label"] == "黄"
    assert "身份描述" in item["message"]
    assert "换装时脸不会飘" in item["message"]


def test_review_flags_moss_unquoted_text(scratch: Path) -> None:
    """MOSS 图内文字没加引号 → unquoted_text 的黄字。"""
    dataset = _dataset_with_images(scratch, ("a.png",))
    (dataset / "a.txt").write_bytes(
        b"mytoken, a poster on a brick wall, "
        b"In-image text: SALE 50% OFF, red and white palette\n"
    )

    response = CLIENT.post(
        "/review",
        data={"dataset_dir": str(dataset), "model": "moss_image", "kind": "text_layout", "trigger": "mytoken"},
    )

    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["code"] == "unquoted_text"
    assert item["severity"] == "warn"
    assert "未加引号" in item["message"]


def test_review_keeps_missing_trigger_red_on_krea(scratch: Path) -> None:
    """review 会把文字送一遍 styler；不许因此把缺触发词这条红错盖掉。"""
    dataset = _dataset_with_images(scratch, ("a.png",))
    (dataset / "a.txt").write_bytes(b"a woman with long black hair, smiling in a cafe\n")

    response = CLIENT.post(
        "/review",
        data={"dataset_dir": str(dataset), "model": "krea2", "kind": "character", "trigger": "mytoken"},
    )

    assert response.status_code == 200, response.text
    item = response.json()["items"][0]
    assert item["code"] == "missing_trigger"
    assert item["severity"] == "error"
