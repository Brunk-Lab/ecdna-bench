"""Tests for ecdna_bench.roi (CPU, no data, no released weights needed)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ecdna_bench.roi import N_PARAMS_RELEASED, UNet
from ecdna_bench.roi import infer, io as roi_io


def test_released_architecture_parameter_count():
    model = UNet(4)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == N_PARAMS_RELEASED
    assert model.enc1.skip.weight.shape[1] == 4


def test_forward_keeps_size_for_odd_shapes():
    model = UNet(4, base=8).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 4, 37, 45))
    assert out.shape == (1, 1, 37, 45)


def test_preprocess_range_and_shape():
    img = np.random.default_rng(0).integers(0, 256, (64, 80, 4), dtype=np.uint8)
    x = infer.preprocess(img, (32, 40))
    assert x.shape == (4, 32, 40) and x.dtype == np.float32
    assert infer._LUT[0] == -1.0 and infer._LUT[255] == 1.0


def test_preprocess_matches_albumentations_208():
    A = pytest.importorskip("albumentations")
    if A.__version__ != "2.0.8":
        pytest.skip("reference is albumentations 2.0.8")
    import cv2
    from albumentations.pytorch import ToTensorV2
    img = np.random.default_rng(1).integers(0, 256, (300, 360, 4), dtype=np.uint8)
    ref = A.Compose([A.Resize(150, 180, interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
                     A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToTensorV2()])(image=img)["image"]
    assert np.array_equal(ref.numpy(), infer.preprocess(img, (150, 180)))


def test_postprocess_keeps_center_component_and_fills_holes():
    probs = np.zeros((200, 240), np.float32)
    probs[60:140, 70:170] = 1.0          # centre blob with a hole
    probs[95:105, 115:125] = 0.0
    probs[5:40, 5:40] = 1.0              # corner blob, not under the centre
    mask = infer.postprocess_mask(probs, threshold=0.4)
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[100, 120] == 255          # hole filled
    assert mask[20, 20] == 0              # corner component dropped


def test_postprocess_keeps_all_when_center_is_background():
    probs = np.zeros((200, 240), np.float32)
    probs[5:60, 5:60] = 1.0
    probs[140:195, 180:235] = 1.0
    mask = infer.postprocess_mask(probs)
    assert mask[30, 30] == 255 and mask[170, 200] == 255


def test_io_detects_png_bytes_under_tif_name(tmp_path):
    import cv2
    dapi = np.random.default_rng(2).integers(0, 256, (20, 30), dtype=np.uint8)
    p = tmp_path / "x.tif"
    ok, buf = cv2.imencode(".png", dapi)
    p.write_bytes(buf.tobytes())
    assert not roi_io.is_tiff(p)
    assert np.array_equal(roi_io.read_dapi(p), dapi)


def test_predict_folder_end_to_end(tmp_path):
    import cv2
    rng = np.random.default_rng(3)
    (tmp_path / "rgb").mkdir(); (tmp_path / "dapi").mkdir()
    for uid in ("a_1", "b_2"):
        rgb = rng.integers(0, 256, (64, 72, 3), dtype=np.uint8)
        cv2.imwrite(str(tmp_path / "rgb" / f"{uid}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(tmp_path / "dapi" / f"{uid}.png"), rng.integers(0, 256, (64, 72), dtype=np.uint8))
    torch.manual_seed(0)
    ckpt = tmp_path / "w.pth"
    torch.save(UNet(4).state_dict(), ckpt)
    n = infer.predict_folder(tmp_path / "rgb", tmp_path / "dapi", ckpt, tmp_path / "out",
                             device="cpu", train_size=(32, 32), save_size=(64, 72))
    assert n == 2
    m = cv2.imread(str(tmp_path / "out" / "a_1.png"), cv2.IMREAD_GRAYSCALE)
    assert m.shape == (64, 72) and set(np.unique(m)) <= {0, 255}
    assert infer.predict_folder(tmp_path / "rgb", tmp_path / "dapi", ckpt, tmp_path / "out",
                                device="cpu", train_size=(32, 32), save_size=(64, 72)) == 0


def test_load_model_is_strict(tmp_path):
    ckpt = tmp_path / "w3.pth"
    torch.save(UNet(3).state_dict(), ckpt)
    with pytest.raises(RuntimeError):
        infer.load_model(ckpt, in_ch=4)


def test_train_smoke(tmp_path):
    pytest.importorskip("albumentations")
    import cv2
    from ecdna_bench.roi.train import train_roi
    rng = np.random.default_rng(4)
    for d in ("rgb", "dapi", "roi", "splits"):
        (tmp_path / d).mkdir()
    ids = [f"img_{i}" for i in range(4)]
    for uid in ids:
        cv2.imwrite(str(tmp_path / "rgb" / f"{uid}.png"), rng.integers(0, 256, (48, 56, 3), dtype=np.uint8))
        cv2.imwrite(str(tmp_path / "dapi" / f"{uid}.png"), rng.integers(0, 256, (48, 56), dtype=np.uint8))
        roi = np.zeros((48, 56), np.uint8); roi[10:40, 10:45] = 255
        cv2.imwrite(str(tmp_path / "roi" / f"{uid}.png"), roi)
    (tmp_path / "splits" / "train_ids.csv").write_text("unique_id\n" + "\n".join(ids[:2]) + "\n")
    (tmp_path / "splits" / "val_ids.csv").write_text("unique_id\n" + "\n".join(ids[2:]) + "\n")
    run = train_roi(tmp_path / "rgb", tmp_path / "dapi", tmp_path / "roi", tmp_path / "splits",
                    tmp_path / "runs", name="smoke", img_size=(32, 32), accum_steps=1,
                    num_epochs=1, device="cpu", num_cpus=1)
    assert (run / "best_checkpoint.pth").is_file() and (run / "final_checkpoint.pth").is_file()
    lines = (run / "train_history.csv").read_text().splitlines()
    assert len(lines) == 2 and lines[0].startswith("epoch,")
    infer.load_model(run / "best_checkpoint.pth", in_ch=4)
