import os
import pathlib

from adtool.utils.filetype_converter.filetype_converter import (
    convert_image,
    convert_video,
    is_mp4,
    is_png,
)


def setup_function(function):
    global TEST_ARTIFACTS
    file_path = str(pathlib.Path(__file__).parent.resolve())
    asset_folder = os.path.join(file_path, "assets")

    img_path = asset_folder + "/img.png"
    vid_path = asset_folder + "/vid.mp4"
    doc_path = asset_folder + "/doc.docx"
    mov_path = asset_folder + "/mkv.mkv"
    with open(img_path, "rb") as f:
        img = f.read()
    with open(vid_path, "rb") as f:
        vid = f.read()
    with open(doc_path, "rb") as f:
        doc = f.read()
    with open(mov_path, "rb") as f:
        mkv = f.read()
    TEST_ARTIFACTS = {"img": img, "vid": vid, "doc": doc, "mkv": mkv}


def teardown_function(function):
    pass


def test_is_mp4():
    assert is_mp4(TEST_ARTIFACTS["vid"])
    assert not is_mp4(TEST_ARTIFACTS["img"])
    assert not is_mp4(TEST_ARTIFACTS["doc"])
    assert not is_mp4(TEST_ARTIFACTS["mkv"])


def test_is_png():
    assert is_png(TEST_ARTIFACTS["img"])
    assert not is_png(TEST_ARTIFACTS["vid"])
    assert not is_png(TEST_ARTIFACTS["doc"])
    assert not is_png(TEST_ARTIFACTS["mkv"])


def test_convert_image():
    png = convert_image(TEST_ARTIFACTS["img"])
    assert is_png(png)


def test_convert_video():
    vid = convert_video(TEST_ARTIFACTS["vid"])
    assert is_mp4(vid)
    # assert that the video is the same as the original
    assert vid == TEST_ARTIFACTS["vid"]

    # FIXME: error with doing the imread for some reason
    mkv = convert_video(TEST_ARTIFACTS["mkv"])
    assert is_mp4(mkv)
    # assert that the video is not the same as the original
    # due to nontrivial conversion
    assert mkv != TEST_ARTIFACTS["mkv"]
