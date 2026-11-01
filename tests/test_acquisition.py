import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import stat
import threading
import zipfile
import pytest
from ascii_flux.acquisition import (default_data_root, download_file, extract_zip,
                                    find_annotations, load_prepared, prepare_ytvis2019, DOWNLOAD_USER_AGENT)


def zip_bytes(entries):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return output.getvalue()


def test_completed_extraction_survives_transient_windows_rename_failure(tmp_path, monkeypatch):
    import ascii_flux.acquisition as acquisition
    archive = tmp_path/"media.zip"
    archive.write_bytes(zip_bytes({"video/frame.jpg": b"analytical cache fixture"}))
    target = tmp_path/"frames"
    real_replace = acquisition.os.replace
    attempts = []
    def locked(source, destination):
        if Path(source).is_dir():
            attempts.append(1)
            raise PermissionError("analytical simulation of antivirus directory lock")
        return real_replace(source, destination)
    monkeypatch.setattr(acquisition.os, "replace", locked)
    monkeypatch.setattr(acquisition.time, "sleep", lambda _: None)
    with pytest.raises(PermissionError):
        extract_zip(archive, target, durable_files=False)
    assert len(attempts) == 15 and not target.exists()
    assert list(tmp_path.glob("frames.*/.extracted.json"))
    monkeypatch.setattr(acquisition.os, "replace", real_replace)
    # Resume promotes the already CRC-verified extraction without extracting again.
    monkeypatch.setattr(acquisition.zipfile, "ZipFile", lambda *_: pytest.fail("unnecessary repeated extraction"))
    assert extract_zip(archive, target) == target.resolve()
    assert (target/"video/frame.jpg").read_bytes() == b"analytical cache fixture"


def source_template():
    root = Path(__file__).resolve().parents[1]
    source = json.loads((root/"configs/ytvis_source.json").read_text(encoding="utf-8"))
    source["synthetic"] = True  # Analytical adapter test, never actual YouTube data.
    return source


def payloads():
    # Tiny source-like fixture; no semantics or D1 result is claimed.
    annotation = {"videos": [{"id": 1, "width": 2, "height": 2,
                               "file_names": ["video-a/00000.jpg", "video-a/00005.jpg"]}],
                  "annotations": [{"id": 1, "video_id": 1, "category_id": 1,
                                    "segmentations": [{"size": [2, 2], "counts": [0, 4]}]*2}],
                  "categories": [{"id": 1, "name": "fixture-only"}]}
    from PIL import Image
    image = io.BytesIO()
    Image.new("RGB", (2, 2), (255, 0, 0)).save(image, format="JPEG")
    images = zip_bytes({f"train/JPEGImages/video-a/{name}.jpg": image.getvalue() for name in ("00000", "00005")})
    return json.dumps(annotation).encode(), images


def test_default_paths_follow_execution_environment(tmp_path):
    project = tmp_path/"project"
    content = tmp_path/"content"
    assert default_data_root(project, content) == project/"data"
    content.mkdir()
    assert default_data_root(project, content) == content/"ascii_flux_data"
    (content/"drive/MyDrive").mkdir(parents=True)
    assert default_data_root(project, content) == content/"drive/MyDrive/ascii_flux"


def test_download_prepare_and_reuse_with_real_http_client(tmp_path):
    pytest.importorskip("gdown")
    annotations, images = payloads()
    calls = []
    user_agents = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            calls.append(self.path)
            user_agent = self.headers.get("User-Agent", "")
            user_agents.append(user_agent)
            # Reproduce Kaggle returning HTTP 200 HTML to gdown's default
            # browser identity; run the actual gdown HTTP client in this test.
            body = (annotations if self.path == "/train.json" else images) if user_agent == DOWNLOAD_USER_AGENT else b"<html>Not an archive</html>"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        ready = prepare_ytvis2019(tmp_path, base+"/train.json", base+"/train.zip", source_template())
        assert ready["source_fps"] == 6 and ready["listed_frames"] == 2
        assert ready["source"]["synthetic"]
        assert Path(ready["frames_root"]).name == "JPEGImages"
        assert load_prepared(tmp_path) == ready
        assert prepare_ytvis2019(tmp_path, base+"/train.json", base+"/train.zip", source_template()) == ready
        assert calls == ["/train.json", "/train.zip"]
        assert user_agents == [DOWNLOAD_USER_AGENT, DOWNLOAD_USER_AGENT]
        with pytest.raises(ValueError, match="incompatible"):
            download_file(base+"/other", tmp_path/"ytvis-2019/downloads/train.zip", "frames")
        Path(ready["annotations"]).write_text("{}")
        with pytest.raises(ValueError, match="modifiées"):
            load_prepared(tmp_path)
    finally:
        server.shutdown()
        server.server_close()
        worker.join(2)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "C:/escape", "..\\escape", "alias./file"])
def test_zip_cannot_escape_workspace_or_alias_paths(tmp_path, name):
    archive = tmp_path/"unsafe.zip"
    archive.write_bytes(zip_bytes({name: b"bad"}))
    with pytest.raises(ValueError):
        extract_zip(archive, tmp_path/"extracted")
    assert not (tmp_path/"extracted").exists()
    assert not list(tmp_path.glob("extracted.*"))


def test_rejects_symlinks_in_zip(tmp_path):
    path = tmp_path/"links.zip"
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../outside")
    with pytest.raises(ValueError, match="symbolique"):
        extract_zip(path, tmp_path/"files")


def test_html_download_is_not_published_as_dataset(tmp_path):
    def downloader(**kwargs):
        Path(kwargs["output"]).write_text("<html>Sign in</html>")
        return kwargs["output"]

    for kind in ("frames", "annotations"):
        target = tmp_path/kind
        with pytest.raises(ValueError):
            download_file("https://example.invalid/fixture", target, kind, downloader)
        assert not target.exists()
    with pytest.raises(ValueError, match="fichier"):
        download_file("https://drive.google.com/drive/folders/example", tmp_path/"bad", "frames", downloader)


def test_annotation_zip_discovery_and_incomplete_frames(tmp_path):
    from ascii_flux.acquisition import find_frames_root
    annotations, images = payloads()
    source = tmp_path/"annotations.zip"
    source.write_bytes(zip_bytes({"annotations/train.json": annotations, "valid.json": b'{"videos": []}'}))
    ann_path, data = find_annotations(extract_zip(source, tmp_path/"annotations"))
    assert ann_path.name == "train.json"
    frames = tmp_path/"frames.zip"
    frames.write_bytes(images)
    frames_root, _ = find_frames_root(extract_zip(frames, tmp_path/"images"), data)
    (frames_root/"video-a/00005.jpg").unlink()
    with pytest.raises(ValueError, match="incomplète"):
        find_frames_root(tmp_path/"images", data)

