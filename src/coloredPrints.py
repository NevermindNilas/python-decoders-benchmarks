from enum import IntEnum

class ColorCode(IntEnum):
    lightgreen = 92
    lightcyan = 96

def _colorSTRTemplate(color: ColorCode) -> str:
    return "\033[%dm{}\033[00m" % (color.value)

def lightgreen(*values: object) -> str:
    return _colorSTRTemplate(ColorCode.lightgreen).format(values[0])

def lightcyan(*values: object) -> str:
    return _colorSTRTemplate(ColorCode.lightcyan).format(values[0])