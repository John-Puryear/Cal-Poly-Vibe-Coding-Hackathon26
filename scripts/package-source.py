from pathlib import Path
import zipfile
project=Path(__file__).resolve().parents[1]
destination=project.parent/'Morphogenesis-Lab-source.zip'
with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
    for item in sorted(project.rglob('*')):
        relative=item.relative_to(project)
        if not item.is_file() or any(p in ('.git','node_modules','__pycache__') for p in relative.parts):continue
        if item.name.endswith('fixture.json') or item.name.startswith('.env'):continue
        archive.write(item,Path('morphogenesis-site')/relative)
print(destination)
