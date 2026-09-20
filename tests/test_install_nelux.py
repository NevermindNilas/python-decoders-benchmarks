import unittest

from packaging.tags import Tag

from src.install_nelux import selectWheel


class NeluxWheelTests(unittest.TestCase):
    def setUp(self):
        self.assets = [{"name": name, "browser_download_url": name} for name in [
            "nelux-0.19.0-214torch-cp313-cp313-manylinux_2_28_x86_64.whl",
            "nelux-0.19.0-214torch-cp314-cp314-manylinux_2_28_x86_64.whl",
            "nelux-0.19.0-214torch-cp314-cp314-win_amd64.whl",
            "nelux-0.19.0-213torch-cp314-cp314-manylinux_2_28_x86_64.whl",
        ]]
        self.tags = {Tag("cp314", "cp314", "manylinux_2_28_x86_64")}

    def test_selects_python_platform_and_torch_abi(self):
        self.assertEqual(selectWheel(self.assets, "2.14.0+cu130", self.tags), self.assets[1]["name"])

    def test_rejects_incompatible_torch_instead_of_installing_broken_wheel(self):
        with self.assertRaisesRegex(RuntimeError, "found 0"):
            selectWheel(self.assets, "2.15.0", self.tags)
