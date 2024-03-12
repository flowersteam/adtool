from io import BytesIO

import filetype
import imageio.v3 as iio


def is_mp4(blob: bytes) -> bool:
    kind = filetype.guess(blob)
    return kind.mime == "video/mp4"


def is_png(blob: bytes) -> bool:
    kind = filetype.guess(blob)
    return kind.mime == "image/png"


def convert_image(blob: bytes) -> bytes:
    ndarray = iio.imread(blob)
    bbuf = BytesIO()
    iio.imwrite(uri=bbuf, image=ndarray, extension=".png")
    return bbuf.getvalue()


def convert_video(blob: bytes) -> bytes:
    # there's actually no impl difference because we just call imageio,
    # although there is a short circuit if the file is already an mp4
    if not is_mp4(blob):
        ndarray = iio.imread(blob)
        bbuf = BytesIO()
        iio.imwrite(uri=bbuf, image=ndarray, codec="h264", extension=".mp4")
        blob = bbuf.getvalue()
    return blob
