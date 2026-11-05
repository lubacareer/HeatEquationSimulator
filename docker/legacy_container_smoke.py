"""Exercise the real legacy GUI and non-root container filesystem boundary."""
from __future__ import annotations

import errno
import faulthandler
import json
import os
from pathlib import Path
import tempfile


def main() -> None:
    assert os.getuid() == os.geteuid() == 10001, "Container must run as its non-root user"
    assert os.getgid() == os.getegid() == 10001, "Container must use its unprivileged group"
    assert Path.cwd() == Path("/data"), "Application outputs must start in /data"
    source = Path("/opt/heat-equation/interface3.py")
    assert source.stat().st_uid == 0, "Application source must remain root-owned"
    for directory in source.parents:
        assert directory.stat().st_uid == 0, f"Source directory must remain root-owned: {directory}"
        assert not os.access(directory, os.W_OK), f"Source directory is writable: {directory}"
    try:
        with source.open("r+"):
            pass
    except OSError as error:
        if error.errno not in (errno.EACCES, errno.EPERM, errno.EROFS):
            raise
    else:
        raise AssertionError("The runtime user can modify application source")
    with tempfile.TemporaryDirectory(prefix="container-smoke-", dir="/data") as directory:
        output = Path(directory) / "result.txt"
        output.write_text("writable data", encoding="utf-8")
        assert output.read_text(encoding="utf-8") == "writable data"

    import tkinter as tk
    from interface3 import App

    application = App()
    try:
        application.root.update()
        assert application.root.winfo_viewable(), "Legacy Tk window did not render"
        title = application.root.title()
        assert title == "Heat Conduction Dashboard"
        application.root.after(200, application.on_closing)
        application.run()
    finally:
        try:
            application.root.destroy()
        except tk.TclError:
            pass
    print(json.dumps({"passed": True, "uid": os.geteuid(), "gid": os.getegid(),
                      "gui": title, "writable_data": True, "protected_source": True}))


if __name__ == "__main__":
    # A broken GUI lifecycle must fail this check instead of hanging its caller.
    faulthandler.dump_traceback_later(30, exit=True)
    try:
        main()
    finally:
        faulthandler.cancel_dump_traceback_later()
