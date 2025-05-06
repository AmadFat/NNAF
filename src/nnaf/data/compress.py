from PIL import Image
import zlib
import io
import lzma
import zstd

@profile
def zlib_bytes_compress(
    data_bytes: bytes,
    compress_level: int = -1,
):
    return zlib.compress(data_bytes, level=compress_level)

@profile
def lzma_bytes_compress(
    data_bytes: bytes,
    compress_level: int = 9,
):
    return lzma.compress(data_bytes)

def zlib_bytes_decompress(
    compressed_img_bytes: bytes,
):
    img_bytes = zlib.decompress(compressed_img_bytes)
    img = Image.open(io.BytesIO(img_bytes))
    return img

@profile
def zstd_bytes_compress(
    data_bytes: bytes,
    compress_level: int = 3,
):
    return zstd.compress(data_bytes, compress_level)