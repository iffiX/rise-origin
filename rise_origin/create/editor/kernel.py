# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Muhan Li (iffi/iffiX)
"""The editing kernel: a plain Jupyter kernel with the create namespace warm.

The persistent editing process is a kernel, not a server: the authoring API
is rich Python, and an agent or a human notebook attaches over the standard
protocol for execute, interrupt, and rich in-band outputs. Off by default,
started deliberately, localhost only; a kernel is arbitrary code execution
and should look like it.

Run as ``python -m rise_origin.create.editor.kernel``; the connection file
path prints on start.
"""

__all__ = ["KernelHost"]


class KernelHost:
    """Starts an ipykernel preloaded with the create namespace."""

    @staticmethod
    def main() -> None:
        """Launch the kernel and print its connection file path."""
        from ipykernel.kernelapp import IPKernelApp

        app = IPKernelApp.instance()
        app.ip = "127.0.0.1"
        app.initialize([])
        print(f"rise create kernel: {app.abs_connection_file}", flush=True)
        app.shell.run_cell(
            "from rise_origin import create as cr\n"
            "import rise_origin.create.modules\n"
        )
        app.start()


if __name__ == "__main__":
    KernelHost.main()
