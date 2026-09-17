"""Share-link support of scripts/fetch_bia_subset.py, against a local mock server."""
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "fetch_bia_subset.py"
KEY = "sharekey-123"
SECRET = "privpath-XYZ"


def _load():
    spec = importlib.util.spec_from_file_location("fetch_bia_subset", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["fetch_bia_subset"] = mod   # dataclasses need the module registered
    spec.loader.exec_module(mod)
    return mod


IMAGES_TSV = (
    "Files\tData Type\tSplit\tSubset\tCell Line\tsource_image\n"
    "images/rgb/img_a.tif\tRGB image\ttest\tbenchmark\tSNU16\t\n"
    "images/index.csv\tIndex table\t\t\t\t\n"
)
GT_TSV = "Files\tAnnotation Type\tsource_image\nimages/gt_mask/img_a.png\tmask\timages/rgb/img_a.tif\n"
BODIES = {
    "filelist_images.tsv": IMAGES_TSV.encode(),
    "filelist_gt.tsv": GT_TSV.encode(),
    "images/rgb/img_a.tif": b"RGB" * 10,
    "images/index.csv": b"unique_id\nimg_a\n",
    "images/gt_mask/img_a.png": b"PNG" * 5,
}


@pytest.fixture
def server():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _body(self):
            if self.path == f"/api/info?key={KEY}":
                host = f"http://127.0.0.1:{self.server.server_port}"
                return json.dumps({"httpLink": f"{host}/.private/30/{SECRET}/S-BIAD4097"}).encode()
            prefix = f"/.private/30/{SECRET}/S-BIAD4097/Files/"
            if self.path.startswith(prefix):
                return BODIES.get(self.path[len(prefix):])
            return None

        def do_GET(self):
            b = self._body()
            self.send_response(200 if b is not None else 404)
            self.send_header("Content-Length", str(len(b or b"")))
            self.end_headers()
            self.wfile.write(b or b"")

        def do_HEAD(self):
            b = self._body()
            self.send_response(200 if b is not None else 404)
            self.send_header("Content-Length", str(len(b or b"")))
            self.end_headers()

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_share_link_key_detection():
    m = _load()
    assert m.share_link_key("https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4097?key=abc") == "abc"
    assert m.share_link_key("https://ftp.ebi.ac.uk/pub/x/Files") is None
    assert m.share_link_key("https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4097") is None


def test_not_a_url_is_a_clear_error(monkeypatch):
    m = _load()
    monkeypatch.delenv("ECDNA_BIA_BASE_URL", raising=False)
    with pytest.raises(SystemExit, match="not a web address"):
        m.resolve_base_url("<line 44 address>")


def test_files_url_is_kept(monkeypatch):
    m = _load()
    assert m.resolve_base_url("https://example.org/S-BIAD4097/Files/") == "https://example.org/S-BIAD4097/Files"


def test_share_link_downloads_and_hides_key(server, tmp_path, monkeypatch, capsys):
    m = _load()
    monkeypatch.setattr(m, "INFO_API", f"{server}/api/info")
    monkeypatch.setenv("ECDNA_BIA_BASE_URL", f"{server}/biostudies/bioimages/studies/S-BIAD4097?key={KEY}")
    rc = m.main(["--out", str(tmp_path), "--split", "test", "--sections", "images,gt",
                 "--workers", "1", "--retries", "1"])
    out = capsys.readouterr()
    assert rc == 0, out
    assert (tmp_path / "images/rgb/img_a.tif").read_bytes() == BODIES["images/rgb/img_a.tif"]
    assert (tmp_path / "images/gt_mask/img_a.png").is_file()
    assert "all selected files are present" in out.out
    for text in (out.out, out.err, (tmp_path / "download_log.tsv").read_text()):
        assert KEY not in text and SECRET not in text


def test_bad_share_key_is_a_clear_error(server, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, "INFO_API", f"{server}/api/info")
    with pytest.raises(SystemExit, match="refused the share link"):
        m.resolve_base_url(f"{server}/biostudies/bioimages/studies/S-BIAD4097?key=wrong")
