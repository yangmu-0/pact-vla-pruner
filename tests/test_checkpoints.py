from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from pact_eval.checkpoints import shadow_checkpoint


class ShadowTests(unittest.TestCase):
    def test_mutable_metadata_is_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            source=Path(tmp)/'model'; source.mkdir()
            out=Path(tmp)/'attempt'; out.mkdir()
            (source/'config.json').write_text('{"original":true}')
            (source/'modeling_prismatic.py').write_text('ORIGINAL=True\n')
            shadow=shadow_checkpoint(source,out)
            (shadow/'config.json').write_text('{"changed":true}')
            (shadow/'modeling_prismatic.py').write_text('ORIGINAL=False\n')
            self.assertEqual((source/'config.json').read_text(),'{"original":true}')
            self.assertEqual((source/'modeling_prismatic.py').read_text(),'ORIGINAL=True\n')
            self.assertEqual(shadow.name,source.name)
            with self.assertRaises(FileExistsError):
                shadow_checkpoint(source,out)

if __name__=='__main__':
    unittest.main()
