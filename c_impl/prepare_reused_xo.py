#!/usr/bin/env python3
"""Stage an identity-checked XO and its original HLS reports for a fresh link."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tarfile


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    xo, reference, source, build, hls_freq = sys.argv[1:]
    xo, reference, source, build = map(Path, (xo, reference, source, build))
    manifest = {}
    xo_entry = None
    for line in (reference / 'xo_manifest.sha256').read_text().splitlines():
        digest, name = line.split(None, 1)
        manifest[Path(name).name] = digest
        if name.endswith('/gdn_forward.xo'):
            xo_entry = name
    if not xo_entry or not re.search(r'\.h' + re.escape(hls_freq) + r'\.', xo_entry):
        raise ValueError('Reference XO HLS clock does not match this build')
    for name in ('gdn_model.cpp', 'gdn_model.h', 'hls_gdn_forward.tcl'):
        if sha(source / name) != manifest[name]:
            raise ValueError('XO/source mismatch: ' + name)
    if sha(xo) != manifest['gdn_forward.xo']:
        raise ValueError('XO hash mismatch')
    for gate in ('xo_gate.exit', 'state_address_gate.exit', 'native_gate.exit'):
        if (reference / gate).read_text().strip() != '0':
            raise ValueError('Reference gate did not pass: ' + gate)
    summary = json.loads((reference / 'xo_gate_summary.json').read_text())
    if summary.get('failures') != [] or summary.get('cluster_report_count') != 16:
        raise ValueError('Reference HLS architecture gate invalid')
    archive = reference / 'build_diagnostics.tar.gz'
    build.mkdir(parents=True, exist_ok=True)
    count = 0
    # Restore reports only, not RTL synthesis, placement, routing or IP caches.
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle:
            parts = Path(member.name).parts
            if (not member.isfile() or len(parts) < 3 or parts[1] != '_x_compile'
                    or Path(member.name).suffix not in ('.xml', '.rpt')):
                continue
            if '..' in parts or Path(member.name).is_absolute():
                raise ValueError('Unsafe report archive path')
            destination = build.joinpath(*parts[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            with bundle.extractfile(member) as inp, destination.open('wb') as out:
                shutil.copyfileobj(inp, out)
            count += 1
    if not list(build.rglob('gdn_forward_csynth.xml')):
        raise ValueError('Reference archive lacks integrated HLS XML')
    shutil.copyfile(xo, build / 'gdn_forward.xo')
    with (build / 'xo_manifest.sha256').open('w') as out:
        for name in ('gdn_model.cpp', 'gdn_model.h', 'hls_gdn_forward.tcl'):
            out.write('{}  {}\n'.format(manifest[name], source / name))
        out.write('{}  {}\n'.format(manifest['gdn_forward.xo'], build / 'gdn_forward.xo'))
    record = dict(xo_sha256=manifest['gdn_forward.xo'], reference=str(reference),
                  reports_archive_sha256=sha(archive), restored_reports=count,
                  hls_mhz=hls_freq, fresh_vivado_implementation=True)
    (build / 'xo_reuse.json').write_text(json.dumps(record, indent=2) + '\n')
    print('REUSE_XO_IDENTITY_PASS', json.dumps(record))


if __name__ == '__main__':
    main()
