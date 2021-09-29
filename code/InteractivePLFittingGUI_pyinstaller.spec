# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

added_files = [
	('./files/conv2Dlookup.pkl', 'files'),
	('./files/help.png', 'files'),
	('./files/config_default.ini', 'files'),
	('./files/logo.ico', 'files'),
	('./files/nk_air.csv', 'files'),
	('./files/nk_CIGS.csv', 'files'),
	('./files/nk_Mo.csv', 'files'),
	]


a = Analysis(['InteractivePLFittingGUI.py'],
             binaries=[],
             datas=added_files,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='InteractivePLFittingGUI',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          upx_exclude=[],
          runtime_tmpdir=None,
          console=False,
	  icon='./files/logo.ico')
