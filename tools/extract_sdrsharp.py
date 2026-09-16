"""Extract build references from the exact .NET single-file SDR# executable.
Format: dotnet/runtime Microsoft.NET.HostModel/Bundle/{Manifest,FileEntry}.cs.
References are local build inputs, never redistributed with the plugin.
"""
from pathlib import Path
import hashlib
import json
import struct
import sys
import zlib


def extract(source, destination):
    data = Path(source).read_bytes()
    # Official .NET apphost bundle marker, SHA-256 of '.net core bundle'.
    marker = bytes.fromhex('8b1202b96a612038727b930214d7a03213f5b9e6efae3318ee3b2dce24b36aae')
    at = data.find(marker)
    if at < 8:
        raise ValueError('Not a supported .NET bundle')
    pos = struct.unpack_from('<q', data, at - 8)[0]

    def read(fmt):
        nonlocal pos
        result = struct.unpack_from(fmt, data, pos)
        pos += struct.calcsize(fmt)
        return result

    def string():
        nonlocal pos
        length = shift = 0
        while True:
            byte = data[pos]; pos += 1
            length |= (byte & 127) << shift
            if byte < 128:
                break
            shift += 7
            if shift > 28:
                raise ValueError('Invalid bundle string')
        value = data[pos:pos + length].decode('utf-8'); pos += length
        return value

    major, minor, count = read('<IIi')
    if major not in (2, 6):
        raise ValueError(f'Unsupported bundle version {major}.{minor}')
    bundle_id = string()
    read('<qqqqQ')
    out = Path(destination); out.mkdir(parents=True, exist_ok=True)
    files = {}
    for _ in range(count):
        offset, size = read('<qq')
        compressed = read('<q')[0] if major >= 6 else 0
        kind = read('<B')[0]
        name = string()
        if Path(name).name != name or not name.startswith('SDRSharp'):
            continue
        payload = data[offset:offset + (compressed or size)]
        if compressed:
            payload = zlib.decompress(payload, -15)
        if len(payload) != size:
            raise ValueError('Incorrect extracted size')
        (out / name).write_bytes(payload)
        files[name] = hashlib.sha256(payload).hexdigest()
    manifest = dict(source=str(Path(source).resolve()), source_sha256=hashlib.sha256(data).hexdigest(), bundle_id=bundle_id, files=files)
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    extract(*sys.argv[1:])
