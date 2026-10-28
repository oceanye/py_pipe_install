"""Verify the real default GUI entry separately from explicit historical open."""
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time
import tkinter as tk

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pipe_twin.gui import _start_application

OUTPUT = ROOT/'outputs/startup-selection-20261028'
TAGS = ('assessment-zone', 'axial-band', 'ground-region', 'solver-selection', 'pooled-boundary')


def snapshot(d, name):
    root = d.window
    root.geometry('1120x860+30+30')
    root.deiconify(); root.lift(); root.attributes('-topmost', True)
    root.update(); time.sleep(.3); root.update()
    x, y = root.winfo_rootx(), root.winfo_rooty()
    path = OUTPUT/f'{name}.png'
    # Native screenshot capture keeps this audit free of optional imaging packages.
    target = str(path).replace("'", "''")
    script = f"""Add-Type -AssemblyName System.Drawing
$shot = [System.Drawing.Bitmap]::new({root.winfo_width()}, {root.winfo_height()})
$canvas = [System.Drawing.Graphics]::FromImage($shot)
try {{
    $canvas.CopyFromScreen({x}, {y}, 0, 0, $shot.Size)
    $shot.Save('{target}', [System.Drawing.Imaging.ImageFormat]::Png)
}} finally {{ $canvas.Dispose(); $shot.Dispose() }}
"""
    subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command', script], check=True)
    root.attributes('-topmost', False)
    return {'screenshot': str(path.relative_to(ROOT)), 'message': d.message.get(),
            'tags': {tag: len(d.views['left'].canvas.find_withtag(tag)) for tag in TAGS},
            'scope': d.zone_settings['scope'], 'zone_count': len(d.zone_settings['zones'])}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    resume = ROOT/'outputs/elevation_workbench/last_scene.json'
    source = Path(json.loads(resume.read_text(encoding='utf8'))['manifest_path'])
    protected = [resume, source, source.with_name('report.json'),
                 ROOT/'outputs/measurement_workbench/calibration_current.json']
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    root = tk.Tk(); root.withdraw()
    d = _start_application(root)
    try:
        assert d.last_manifest == source
        result = {'source': str(source.relative_to(ROOT)), 'default_startup': snapshot(d, 'default-startup')}
        assert all(count == 0 for count in result['default_startup']['tags'].values())
        assert d.zone_settings['zones'] == []
        assert 'aggregation_band' not in d.registration_settings
        assert 'ground_region' not in d.registration_settings
        assert d.report is None
        photos = dict(d.image_hashes)
        calibration = d.current_calibration()
        d.load_session(source)
        result['explicit_history'] = snapshot(d, 'explicit-history')
        assert all(result['explicit_history']['tags'][tag] > 0 for tag in TAGS[:3])
        assert d.image_hashes == photos
        assert d.current_calibration() == calibration
        result['photo_hashes_unchanged'] = photos
    finally:
        d.close()
    for name, expected in hashes.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected, name
    result['source_hashes_unchanged'] = hashes
    Path(__file__).with_name('startup-validation.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf8', newline='\n')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
