"""
Windows shim for the Unix-only `fcntl` stdlib module.

nuplan-devkit's map/database code (nuplan/database/maps_db/gpkg_mapsdb.py) does
`import fcntl` at module load, which is pulled in transitively when navsim imports
`nuplan.common.maps...` -- even though the nuScenes training/eval path never actually
uses nuplan maps or file locking. This stub lets those imports resolve on Windows.

Install (Windows only) by copying this file into the env's site-packages, e.g.:
    copy tools\windows_shims\fcntl.py <env>\Lib\site-packages\fcntl.py

Cross-platform-safe: the CPython stdlib `fcntl` (a builtin extension module) is found
BEFORE site-packages on Linux/macOS, so this stub only ever takes effect on Windows.
The functions are no-ops; if any code path actually calls them at runtime you are on a
code path that needs real file locking (i.e. real nuplan map DB access) and should run
on Linux/WSL2 instead.
"""

# Lock operation flags (values match Linux <bits/fcntl-linux.h> for familiarity)
LOCK_SH = 1
LOCK_EX = 2
LOCK_NB = 4
LOCK_UN = 8

# Common fcntl command constants (present so `fcntl.F_*` attribute access won't crash)
F_GETFD = 1
F_SETFD = 2
F_GETFL = 3
F_SETFL = 4
F_GETLK = 5
F_SETLK = 6
F_SETLKW = 7


def flock(fd, operation):  # noqa: D401 - no-op on Windows
    return None


def lockf(fd, cmd, length=0, start=0, whence=0):
    return None


def fcntl(fd, cmd, arg=0):
    return 0


def ioctl(fd, request, arg=0, mutate_flag=True):
    return 0
