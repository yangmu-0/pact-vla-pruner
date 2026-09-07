"""Keep legacy loader code synchronization out of historical checkpoints."""
import hashlib
import json
from pathlib import Path
import shutil


def shadow_checkpoint(source, output):
    source=Path(source).resolve()
    target=Path(output)/'checkpoint_shadow'/source.name
    target.mkdir(parents=True,exist_ok=False)
    files=[]
    for item in sorted(source.iterdir()):
        if not item.is_file() or '.back.' in item.name or item.name.endswith('.bak'):
            continue
        dest=target/item.name
        if item.suffix in ('.safetensors','.pt','.bin','.model'):
            dest.symlink_to(item.resolve())
            files.append(dict(name=item.name,mode='input_link',target=str(item.resolve())))
        else:
            shutil.copy2(item,dest)
            files.append(dict(name=item.name,mode='private_copy',source_sha256=hashlib.sha256(item.read_bytes()).hexdigest()))
    (Path(output)/'checkpoint_shadow.json').write_text(json.dumps(dict(source=str(source),shadow=str(target),files=files),indent=2))
    return target
