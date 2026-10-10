from pathlib import Path
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED

root = Path('/home/js2723/PROJECTS/VUD')
files = sorted((root / 'results').rglob('*.png'))
if not files:
    raise SystemExit('No PNGs found')
stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
archive = root / 'tmp' / f'results_pngs_{stamp}.zip'
with ZipFile(archive, 'x', compression=ZIP_DEFLATED) as bundle:
    for path in files:
        bundle.write(path, path.relative_to(root))
with ZipFile(archive) as bundle:
    assert bundle.testzip() is None
    assert len(bundle.namelist()) == len(files)
print(f'Archive: {archive}', flush=True)
print(f'PNGs: {len(files)}; bytes: {archive.stat().st_size}', flush=True)
