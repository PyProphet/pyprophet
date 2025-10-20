"""
PyInstaller hook for pyopenms package.
Collects native libraries and Qt dependencies required by pyOpenMS.
"""
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules
import os
import sys

# Collect all submodules
hiddenimports = collect_submodules('pyopenms')

# Collect all dynamic libraries (.so on Linux, .dylib on macOS, .dll on Windows)
binaries = collect_dynamic_libs('pyopenms')

# Collect data files (if any)
datas = collect_data_files('pyopenms')

# On macOS, we need to find and include the dependency libraries that pyopenms links to
if sys.platform == 'darwin':
    try:
        import pyopenms
        pyopenms_dir = os.path.dirname(pyopenms.__file__)
        
        # Common dependency library patterns to look for
        lib_patterns = [
            'libboost_*.dylib',
            'libbrotli*.dylib',
            'libicu*.dylib',
            'libQt*.dylib',
            'libOpenMS*.dylib',
            'libsqlite*.dylib',
            'libhdf5*.dylib',
            'libxerces-c*.dylib',
        ]
        
        # Search for libraries in pyopenms directory and parent directories
        import glob
        for pattern in lib_patterns:
            # Check pyopenms directory
            for lib in glob.glob(os.path.join(pyopenms_dir, pattern)):
                binaries.append((lib, 'pyopenms'))
            
            # Check parent directory (common conda location)
            parent_dir = os.path.dirname(pyopenms_dir)
            for lib in glob.glob(os.path.join(parent_dir, pattern)):
                binaries.append((lib, '.'))
            
            # Check site-packages lib directory
            lib_dir = os.path.join(os.path.dirname(parent_dir), 'lib')
            if os.path.exists(lib_dir):
                for lib in glob.glob(os.path.join(lib_dir, pattern)):
                    binaries.append((lib, '.'))
        
        # Also try to find conda environment libraries
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            conda_lib = os.path.join(conda_prefix, 'lib')
            if os.path.exists(conda_lib):
                for pattern in lib_patterns:
                    for lib in glob.glob(os.path.join(conda_lib, pattern)):
                        binaries.append((lib, '.'))
        
        print(f"pyopenms hook: collected {len(binaries)} binary files")
        
    except Exception as e:
        print(f"pyopenms hook warning: {e}")

elif sys.platform.startswith('linux'):
    # Similar logic for Linux .so files
    try:
        import pyopenms
        pyopenms_dir = os.path.dirname(pyopenms.__file__)
        
        lib_patterns = [
            'libboost_*.so*',
            'libbrotli*.so*',
            'libicu*.so*',
            'libQt*.so*',
            'libOpenMS*.so*',
        ]
        
        import glob
        for pattern in lib_patterns:
            for lib in glob.glob(os.path.join(pyopenms_dir, pattern)):
                binaries.append((lib, 'pyopenms'))
                
    except Exception as e:
        print(f"pyopenms hook warning: {e}")
