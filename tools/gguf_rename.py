#!/usr/bin/env python3
"""GGUF renamer for audiocpp conversion inputs (ACE-Step 1.5 SFT path).

Fixes physical tensor names >= 64 chars (GGML_MAX_NAME hard gate) by:
  1. shortening them to unique `_renamed.<i>` names,
  2. appending an `audiocpp.tensor_names` metadata array (string array, one
     entry per tensor in file order) that preserves the ORIGINAL logical names,
     exactly mirroring the converter's own output convention
     (src/framework/assets/tensor_source.cpp, GgufTensorSource) so logical-name
     resolution works through the metadata.

Also splits a named tensor into its own single-tensor GGUF (used to route the
author file's `silence_latent` into its own spec namespace).

Header layout (this ggml fork — verified against external/ggml/src/gguf.cpp):
  magic "GGUF", version u32, n_tensors u64, n_kv u64,
  KV pairs (key string, type u32, value), tensor infos
  (name string, n_dims u32, dims i64*n, type u32, offset u64),
  then data section padded to 32. Header strings are NOT aligned; only the
  data section is 32-aligned (GGUF_DEFAULT_ALIGNMENT).

Values are re-emitted as raw bytes (no float decode/encode).
Every read is sanity-capped — a parse error raises with the file position
instead of producing a MemoryError or silently desyncing.

Usage:
  gguf_rename.py <input.gguf> <output.gguf> [--split-silence <silence.gguf>]
  gguf_rename.py --test            # synthetic round-trip self-test
"""
import struct, sys

KV_TYPES = {  # type -> (size, is_string, is_array)
    0: (1, False, False), 1: (1, False, False), 2: (2, False, False),
    3: (2, False, False), 4: (4, False, False), 5: (4, False, False),
    6: (4, False, False), 7: (1, False, False), 8: (None, True, False),
    9: (None, False, True), 10: (8, False, False), 11: (8, False, False),
    12: (8, False, False), 13: (2, False, False), 14: (2, False, False),
}
_MAX_NAME = 4 * 1024 * 1024  # embedded config/tokenizer strings can be large
_MAX_DIMS = 16
_MAX_ARR = 1_000_000_000  # embedded file data arrays can be tens of MB


def read_str(f):
    raw = f.read(8)
    if len(raw) != 8:
        raise RuntimeError(f'truncated string length at {f.tell()}')
    n = struct.unpack('<Q', raw)[0]
    if n > _MAX_NAME:
        raise RuntimeError(f'implausible string length {n} at {f.tell() - 8}')
    return f.read(n).decode('utf-8', 'replace')


def write_str(b, s):
    payload = struct.pack('<Q', len(s)) + s.encode('utf-8')
    if hasattr(b, 'write'):
        b.write(payload)
    else:
        b += payload


def read_value_raw(f, ktype):
    """Read a KV value, returning (raw_bytes, [string_elems]) — raw bytes
    re-emittable verbatim; string array elements extracted."""
    size, is_str, is_arr = KV_TYPES[ktype]
    if is_str:
        n = struct.unpack('<Q', f.read(8))[0]
        if n > _MAX_NAME:
            raise RuntimeError(f'implausible string length {n} at {f.tell() - 8}')
        data = f.read(n)
        return struct.pack('<Q', n) + data, [data.decode('utf-8', 'replace')]
    if is_arr:
        atype = struct.unpack('<I', f.read(4))[0]
        n = struct.unpack('<Q', f.read(8))[0]
        if n > _MAX_ARR:
            raise RuntimeError(f'implausible array length {n} at {f.tell() - 12}')
        asize, ais_str, _ = KV_TYPES[atype]
        if ais_str:
            raw = [struct.pack('<I', atype), struct.pack('<Q', n)]
            elems = []
            for _ in range(n):
                m = struct.unpack('<Q', f.read(8))[0]
                if m > _MAX_NAME:
                    raise RuntimeError(f'implausible array string length {m} at {f.tell() - 8}')
                d = f.read(m)
                raw.append(struct.pack('<Q', m))
                raw.append(d)
                elems.append(d.decode('utf-8', 'replace'))
            return b''.join(raw), elems
        # non-string arrays: bulk read (avoids O(n^2) bytes concatenation)
        data = f.read(n * asize)
        if len(data) != n * asize:
            raise RuntimeError(f'truncated array value at {f.tell()}')
        return struct.pack('<I', atype) + struct.pack('<Q', n) + data, None
    return f.read(size), None


def parse(path):
    """Returns dict: kvs (list of (key, ktype, raw, str_elems)), infos
    (list of (name, rest_bytes)) where rest = n_dims+dims+type+offset,
    version, n_tensors, data_start, file_size, path."""
    with open(path, 'rb') as f:
        magic = f.read(4)
        if magic != b'GGUF':
            raise RuntimeError(f'not a GGUF file: {path}')
        version = struct.unpack('<I', f.read(4))[0]
        n_tensors = struct.unpack('<Q', f.read(8))[0]
        n_kv = struct.unpack('<Q', f.read(8))[0]
        if n_tensors > 10_000_000 or n_kv > 1_000_000:
            raise RuntimeError(f'implausible counts: n_tensors={n_tensors} n_kv={n_kv}')
        kvs = []
        for _ in range(n_kv):
            key = read_str(f)
            ktype = struct.unpack('<I', f.read(4))[0]
            if ktype not in KV_TYPES:
                raise RuntimeError(f'unknown KV type {ktype} after key {key!r} at {f.tell()}')
            raw, elems = read_value_raw(f, ktype)
            kvs.append((key, ktype, raw, elems))
        infos = []
        for i in range(n_tensors):
            name = read_str(f)
            n_dims = struct.unpack('<I', f.read(4))[0]
            if n_dims > _MAX_DIMS:
                raise RuntimeError(f'implausible n_dims {n_dims} for tensor {i} ({name!r}) at {f.tell() - 4}')
            # rest = dims i64*n + type u32 + offset u64 (n_dims field is above)
            rest = f.read(8 * n_dims + 4 + 8)
            if len(rest) != 8 * n_dims + 4 + 8:
                raise RuntimeError(f'truncated tensor info {i} ({name!r}) at {f.tell()}')
            infos.append((name, n_dims, rest))
        header_end = f.tell()
        data_start = (header_end + 31) & ~31
        f.seek(0, 2)
        file_size = f.tell()
    if data_start > file_size:
        raise RuntimeError(f'data section start {data_start} beyond file size {file_size}')
    return {'version': version, 'n_tensors': n_tensors, 'kvs': kvs,
            'infos': infos, 'data_start': data_start, 'file_size': file_size,
            'path': path}


def build_renamed(parsed, out_path, drop=None, strip=None):
    """Shorten over-64 physical names, append audiocpp.tensor_names, write.

    drop: optional set of tensor names to remove entirely (tensor info AND
    metadata entry) — used to shed a tensor that is also routed through its
    own spec namespace (e.g. the author DiT file's embedded silence_latent,
    which gets a dedicated dit_*_silence_latent input instead).

    strip: optional name prefix to remove from every tensor (e.g. the LM
    component ships Qwen-style `model.`-prefixed names, while the loader
    contract — set by the audio-cpp turbo/base packages — is bare). Stripped
    names are all < 64 and unique, so no metadata is appended: physical names
    ARE the logical names."""
    infos = [(n, nd, r) for n, nd, r in parsed['infos'] if drop is None or n not in drop]
    original_names = [n for n, _, _ in infos]
    if strip:
        stripped = {n: n[len(strip):] if n.startswith(strip) else n for n in original_names}
        assert len(set(stripped.values())) == len(stripped), 'strip produces duplicate names'
        infos = [(stripped[n], nd, r) for n, nd, r in infos]
        original_names = [stripped[n] for n in original_names]
    used = set()
    renamed = {}
    for i, (name, _nd, _rest) in enumerate(infos):
        if len(name) >= 64:
            short = f'_renamed.{i}'
            assert short not in used and len(short) < 64, short
            renamed[name] = short
            name = short
        used.add(name)
    # Compute exact shapes (fixes Serveurperso's stripped leading-1 batch dims).
    exact_ranks, exact_shapes, n_fixes = compute_exact_shapes(infos)
    if n_fixes:
        print(f'inject exact shapes: {n_fixes} tensors needed rank fixes')
    data_len = parsed['file_size'] - parsed['data_start']
    with open(parsed['path'], 'rb') as src, open(out_path, 'wb') as dst:
        dst.write(b'GGUF')
        dst.write(struct.pack('<I', parsed['version']))
        dst.write(struct.pack('<Q', len(infos)))
        dst.write(struct.pack('<Q', len(parsed['kvs']) + 3))  # +names +ranks +shapes
        for key, ktype, raw, _elems in parsed['kvs']:
            write_str(dst, key)
            dst.write(struct.pack('<I', ktype))
            dst.write(raw)
        # appended metadata: audiocpp.tensor_names = string array (file order)
        write_str(dst, 'audiocpp.tensor_names')
        dst.write(struct.pack('<I', 9))                # array
        dst.write(struct.pack('<I', 8))                # of string
        dst.write(struct.pack('<Q', len(original_names)))
        for n in original_names:
            write_str(dst, n)
        # exact shape metadata: fixes rank loss from upstream GGUF writers.
        _write_array_kv(dst, 'audiocpp.tensor_ranks', 5, '<' + 'i' * len(exact_ranks), exact_ranks)
        _write_array_kv(dst, 'audiocpp.tensor_shapes', 11, '<' + 'q' * len(exact_shapes), exact_shapes)
        for name, n_dims, rest in infos:
            if name in renamed:
                name = renamed[name]
            write_str(dst, name)
            dst.write(struct.pack('<I', n_dims))
            dst.write(rest)
        pad = (32 - (dst.tell() % 32)) % 32
        dst.write(b'\0' * pad)
        src.seek(parsed['data_start'])
        remaining = data_len
        while remaining > 0:
            chunk = src.read(min(1 << 20, remaining))
            if not chunk:
                raise RuntimeError('truncated source data section')
            dst.write(chunk)
            remaining -= len(chunk)
    return renamed


# ── exact-shape injection ──────────────────────────────────────────────
# Some upstream GGUF writers (notably Serveurperso's ACE-Step conversions)
# strip leading-1 batch dimensions from tensors like `scale_shift_table`,
# `special_tokens`, `null_condition_emb`, and `silence_latent`. The audiocpp
# converter falls back to `ggml_n_dims()` for inputs without exact-shape KVs,
# and ggml normalizes away trailing-1 dims — so the rank loss propagates to
# the merged package and triggers "tensor shape mismatch" at load time.
#
# Injecting `audiocpp.tensor_ranks` + `audiocpp.tensor_shapes` KVs (the
# authoritative shape source the converter and loader both check first)
# fixes this at the metadata level: the tensor data bytes are unchanged.

_RANK_FIX_SUFFIXES = {
    '.scale_shift_table': 3,
    '.special_tokens': 3,
    '.special_token': 3,       # mutually exclusive with .special_tokens (trailing 's')
}
_RANK_FIX_EXACT = {
    'null_condition_emb': 3,
    'silence_latent': 3,
}

# Some tensors have dims stored in PyTorch (outer-in) order instead of ggml
# (inner-out) order, so the generic reversal produces swapped dims. Override
# with the known-correct PyTorch shape (verified against the audio-cpp turbo
# package, which sourced from safetensors).
_SHAPE_OVERRIDES = {
    'silence_latent': [1, 64, 15000],
}


def _target_rank(name):
    """Return the target rank if this tensor needs a leading-1 batch fix."""
    if name in _RANK_FIX_EXACT:
        return _RANK_FIX_EXACT[name]
    for suffix, rank in _RANK_FIX_SUFFIXES.items():
        if name.endswith(suffix):
            return rank
    return None


def compute_exact_shapes(infos):
    """Given [(name, n_dims, rest_bytes), ...], return (ranks, shapes):
    int32 ranks per tensor and a flat int64 shapes list (PyTorch convention:
    reversed ggml ne[], with leading-1 fixes applied)."""
    ranks = []
    shapes = []
    fixes = 0
    for name, n_dims, rest in infos:
        if name in _SHAPE_OVERRIDES:
            shape = list(_SHAPE_OVERRIDES[name])
            fixes += 1
        else:
            dims = struct.unpack('<' + 'q' * n_dims, rest[:8 * n_dims])
            shape = list(reversed(dims))  # ggml ne[] → PyTorch convention
            target = _target_rank(name)
            if target is not None and len(shape) < target:
                shape = [1] * (target - len(shape)) + shape
                fixes += 1
        ranks.append(len(shape))
        shapes.extend(shape)
    return ranks, shapes, fixes


def _write_array_kv(dst, key, atype, pack_fmt, values):
    """Write a GGUF array KV: key string, type=9 (array), atype, count, data."""
    write_str(dst, key)
    dst.write(struct.pack('<I', 9))               # GGUF_TYPE_ARRAY
    dst.write(struct.pack('<I', atype))           # element type
    dst.write(struct.pack('<Q', len(values)))
    dst.write(struct.pack(pack_fmt, *values))


def build_single_tensor(parsed, tensor_name, out_path):
    """Extract one tensor (by name) into a minimal single-tensor GGUF.

    Byte size by offset delta (data laid out contiguously in file order):
    next tensor's offset - this tensor's offset, or data-section remaining
    for the last tensor. No ggml type table needed."""
    index = next((i for i, (n, _nd, _r) in enumerate(parsed['infos']) if n == tensor_name), None)
    if index is None:
        raise RuntimeError(f'tensor not found: {tensor_name}')
    name, n_dims, rest = parsed['infos'][index]
    offset = struct.unpack('<Q', rest[-8:])[0]
    if index + 1 < len(parsed['infos']):
        next_offset = struct.unpack('<Q', parsed['infos'][index + 1][2][-8:])[0]
        nbytes = next_offset - offset
    else:
        nbytes = (parsed['file_size'] - parsed['data_start']) - offset
    data_pos = parsed['data_start'] + offset
    # Exact shapes for the single tensor (fixes silence_latent rank loss).
    s_infos = [(name, n_dims, rest)]
    exact_ranks, exact_shapes, n_fixes = compute_exact_shapes(s_infos)
    with open(parsed['path'], 'rb') as src, open(out_path, 'wb') as dst:
        dst.write(b'GGUF')
        dst.write(struct.pack('<I', 3))
        dst.write(struct.pack('<Q', 1))     # 1 tensor
        dst.write(struct.pack('<Q', 3))     # 3 KVs: architecture, ranks, shapes
        write_str(dst, 'general.architecture')
        dst.write(struct.pack('<I', 8))  # string
        write_str(dst, 'audiocpp')
        _write_array_kv(dst, 'audiocpp.tensor_ranks', 5, '<' + 'i' * len(exact_ranks), exact_ranks)
        _write_array_kv(dst, 'audiocpp.tensor_shapes', 11, '<' + 'q' * len(exact_shapes), exact_shapes)
        write_str(dst, name)
        dst.write(struct.pack('<I', n_dims))
        dst.write(rest[:-8])             # dims + type
        dst.write(struct.pack('<Q', 0))  # offset 0 (first tensor in new file)
        pad = (32 - (dst.tell() % 32)) % 32
        dst.write(b'\0' * pad)
        src.seek(data_pos)
        data = src.read(nbytes)
        if len(data) != nbytes:
            raise RuntimeError(f'truncated tensor data: {name}')
        dst.write(data)
    return (index, name, nbytes, data_pos)


def logical_names(path, strip_prefix=None):
    """Dump logical tensor names (audiocpp.tensor_names metadata, falling back
    to physical names) — used to compare converter inputs against merged
    audiocpp files (the loader resolves tensors by logical name)."""
    parsed = parse(path)
    meta = next((e for k, t, r, e in parsed['kvs'] if k == 'audiocpp.tensor_names'), None)
    names = meta if meta is not None else [n for n, _, _ in parsed['infos']]
    if strip_prefix:
        names = [n[len(strip_prefix):] if n.startswith(strip_prefix) else n for n in names]
    return names


def selftest():
    """Synthetic round-trip: build a 3-tensor GGUF (one name > 64 chars),
    rename it, verify names, metadata, and byte-identical data."""
    import io, tempfile, os
    buf = io.BytesIO()
    long_name = 'tokenizer.attention_pooler.layers.0.post_attention_layernorm.weight'  # 67
    assert len(long_name) > 64
    names = ['decoder.condition_embedder.bias', long_name, 'silence_latent']
    buf.write(b'GGUF')
    buf.write(struct.pack('<I', 3))
    buf.write(struct.pack('<Q', 3))
    buf.write(struct.pack('<Q', 1))
    write_str(buf, 'general.architecture')
    buf.write(struct.pack('<I', 8))
    write_str(buf, 'audiocpp')
    data = []
    for i, name in enumerate(names):
        write_str(buf, name)
        buf.write(struct.pack('<I', 1))     # n_dims
        buf.write(struct.pack('<q', i + 1))  # dims[0] = i+1
        buf.write(struct.pack('<I', 0))     # type f32
        buf.write(struct.pack('<Q', sum((j + 1) * 4 for j in range(i))))  # offset
        data.append(bytes((i + 1) * 4))
    pad = (32 - (buf.tell() % 32)) % 32
    buf.write(b'\0' * pad)
    for d in data:
        buf.write(d)
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, 'src.gguf')
        out = os.path.join(td, 'out.gguf')
        sil = os.path.join(td, 'silence.gguf')
        with open(src, 'wb') as f:
            f.write(buf.getvalue())
        parsed = parse(src)
        renamed = build_renamed(parsed, out)
        assert list(renamed) == [long_name], renamed
        check = parse(out)
        out_names = [n for n, _, _ in check['infos']]
        assert out_names[0] == names[0] and out_names[1] == '_renamed.1' and out_names[2] == names[2]
        meta = next((e for k, t, r, e in check['kvs'] if k == 'audiocpp.tensor_names'), None)
        assert meta == names, 'metadata must hold original logical names'
        # data section byte-identical
        with open(src, 'rb') as f:
            f.seek(parsed['data_start'])
            src_data = f.read()
        with open(out, 'rb') as f:
            f.seek(check['data_start'])
            out_data = f.read()
        assert src_data == out_data, 'data section changed'
        info = build_single_tensor(parsed, 'silence_latent', sil)
        assert info[1] == 'silence_latent' and info[2] == 12
        s2 = parse(sil)
        assert s2['n_tensors'] == 1 and s2['infos'][0][0] == 'silence_latent'
        with open(sil, 'rb') as f:
            f.seek(s2['data_start'])
            assert f.read(12) == data[2], 'silence data mismatch'
        dropped = os.path.join(td, 'dropped.gguf')
        build_renamed(parsed, dropped, drop={'silence_latent'})
        d2 = parse(dropped)
        assert d2['n_tensors'] == 2 and d2['infos'][0][0] == names[0] and d2['infos'][1][0] == '_renamed.1'
        m2 = next((e for k, t, r, e in d2['kvs'] if k == 'audiocpp.tensor_names'), None)
        assert m2 == names[:2], 'dropped metadata must keep remaining logical names'
        stripped = os.path.join(td, 'stripped.gguf')
        build_renamed(parsed, stripped, strip='tokenizer.')
        s2 = parse(stripped)
        got = [n for n, _, _ in s2['infos']]
        assert got[0] == names[0] and got[1] == 'attention_pooler.layers.0.post_attention_layernorm.weight', got
        assert got[2] == 'silence_latent', got
    print('selftest: PASS (rename metadata + data verbatim + split + drop + strip)')


if __name__ == '__main__':
    if '--test' in sys.argv:
        selftest()
        sys.exit(0)
    if '--logical-names' in sys.argv:
        path = sys.argv[sys.argv.index('--logical-names') + 1]
        strip = None
        if '--strip-prefix' in sys.argv:
            strip = sys.argv[sys.argv.index('--strip-prefix') + 1]
        for n in logical_names(path, strip):
            print(n)
        sys.exit(0)
    args = sys.argv[1:]
    drop = set()
    if '--drop' in args:
        at = args.index('--drop')
        if at + 1 >= len(args):
            print('--drop requires a tensor name')
            sys.exit(2)
        drop.add(args[at + 1])
        del args[at:at + 2]
    strip = None
    if '--strip-prefix' in args:
        at = args.index('--strip-prefix')
        if at + 1 >= len(args):
            print('--strip-prefix requires a prefix string')
            sys.exit(2)
        strip = args[at + 1]
        del args[at:at + 2]
    if len(args) < 2:
        print('usage: gguf_rename.py <input.gguf> <output.gguf> [--split-silence <silence.gguf>] [--drop <tensor-name>] [--strip-prefix <prefix>]')
        sys.exit(2)
    src_path, out_path = args[0], args[1]
    parsed = parse(src_path)
    renamed = build_renamed(parsed, out_path, drop, strip)
    print(f'renamed {len(renamed)} over-64 tensors:')
    for old, new in renamed.items():
        print(f'  {old} ({len(old)}c) -> {new}')
    check = parse(out_path)
    names = [n for n, _, _ in check['infos']]
    assert len(names) == check['n_tensors'], 'tensor count mismatch'
    assert all(len(n) < 64 for n in names), 'output still has over-64 names'
    has_meta = any(k == 'audiocpp.tensor_names' for k, _, _, _ in check['kvs'])
    assert has_meta, 'missing audiocpp.tensor_names metadata'
    print(f'ok: {out_path} tensors={len(names)} over64=0 metadata=present')
    if '--split-silence' in args:
        sil_path = args[args.index('--split-silence') + 1]
        info = build_single_tensor(parsed, 'silence_latent', sil_path)
        print(f'ok: {sil_path} tensor={info[1]} nbytes={info[2]} at={info[3]}')
