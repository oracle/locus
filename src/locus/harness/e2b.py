# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""An E2B sandbox as a workspace.

`E2B <https://e2b.dev>`_ runs sandboxes as a hosted service; its Python
client (``pip install "locus-sdk[e2b]"``, the ``e2b`` package) runs a
command with ``sandbox.commands.run`` and reads and writes a file with
``sandbox.files``, which is exactly
:class:`~locus.harness.session.SessionLike`. :class:`E2BSession` is the
adapter and everything else is
:class:`~locus.harness.session.SessionBackend`.

Two facts about the client shape the adapter: a command that exits
non-zero is reported as an exception that carries the exit code and both
streams, and a command that runs past ``timeout`` is reported as a
timeout exception. Both are turned back into the status-and-output pair
the session protocol wants.

This backend is written against the client's documented surface and
checked with a stand-in in the unit tests; the live test under
``tests/integration`` runs only when ``E2B_API_KEY`` is set. It has not been
run against an E2B account by the maintainers.

The ``e2b`` package is imported only by :meth:`E2BBackend.connect` and
:meth:`E2BBackend.create`. Pass a sandbox of your own to the constructor
and nothing from it is imported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from locus.harness.local import MAX_BACKGROUND
from locus.harness.session import SessionBackend


__all__ = ["E2BBackend", "E2BSession"]

#: The working directory of E2B's default template.
SANDBOX_ROOT = "/home/user"

#: Seconds an internal operation (a file move) may take.
_DEFAULT_TIMEOUT = 60.0


def _is(exc: BaseException, *names: str) -> bool:
    """Whether ``exc``'s class name contains one of ``names``: the client's own types."""
    kind = type(exc).__name__
    return any(name in kind for name in names)


class E2BSession:
    """One E2B sandbox as a :class:`~locus.harness.session.SessionLike`.

    Args:
        sandbox: An ``e2b.Sandbox``, or anything with its ``commands.run``,
            ``files.read``, ``files.write`` and ``kill``.
    """

    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox

    def run(
        self,
        command: str,
        *,
        timeout: float,
        on_output: Callable[[bytes], None] | None = None,
    ) -> tuple[int | None, bytes]:
        """Run ``command`` in the sandbox's shell; its status and its output.

        Standard output and standard error arrive through callbacks as the
        command writes them, so they are joined in that order.
        """
        parts: list[bytes] = []

        def sink(text: str) -> None:
            data = text.encode()
            parts.append(data)
            if on_output is not None:
                on_output(data)

        try:
            result = self.sandbox.commands.run(
                command, timeout=timeout, on_stdout=sink, on_stderr=sink
            )
        except Exception as exc:
            code = getattr(exc, "exit_code", None)
            if isinstance(code, int) and not isinstance(code, bool):
                if not parts:
                    parts.append(_streams(exc))
                return code, b"".join(parts)
            if _is(exc, "Timeout"):
                return None, b"".join(parts)
            raise
        if not parts:
            parts.append(_streams(result))
        code = getattr(result, "exit_code", 0)
        return (code if isinstance(code, int) else 0), b"".join(parts)

    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
        return self.run(command, timeout=timeout)

    def upload_file(self, path: str, data: bytes) -> None:
        self.sandbox.files.write(path, data)

    def download_file(self, path: str) -> bytes:
        try:
            data = self.sandbox.files.read(path, format="bytes")
        except Exception as exc:
            if _is(exc, "NotFound"):
                raise FileNotFoundError(path) from exc
            raise
        return bytes(data)


def _streams(result: Any) -> bytes:
    """Stdout then stderr of a result or an exit exception, when nothing was streamed."""
    out = getattr(result, "stdout", "") or ""
    err = getattr(result, "stderr", "") or ""
    return (str(out) + str(err)).encode()


class E2BBackend(SessionBackend):
    """An E2B sandbox as a workspace. Isolated; rooted at ``/home/user``.

    Args:
        sandbox: An ``e2b.Sandbox`` or anything with the methods
            :class:`E2BSession` names.
        root: The workspace directory inside the sandbox.
        owns_sandbox: Kill the sandbox on :meth:`close`. Set by :meth:`create`.
        env: Variables added to every command's environment.
        max_background: How many background commands may run at once.
        state_dir: Where background commands keep their state.
    """

    def __init__(
        self,
        sandbox: Any,
        *,
        root: str = SANDBOX_ROOT,
        owns_sandbox: bool = False,
        env: Mapping[str, str] | None = None,
        max_background: int = MAX_BACKGROUND,
        state_dir: str = "/tmp/.locus-harness",  # noqa: S108 - inside the sandbox
    ) -> None:
        self._e2b = E2BSession(sandbox)
        sandbox_id = getattr(sandbox, "sandbox_id", None) or "sandbox"
        super().__init__(
            self._e2b,
            root=root,
            label=f"e2b:{sandbox_id}",
            isolated=True,
            state_dir=state_dir,
            env=env,
            max_background=max_background,
        )
        self._owns_sandbox = owns_sandbox

    @property
    def sandbox(self) -> Any:
        """The sandbox every call goes through."""
        return self._e2b.sandbox

    def _exec_script(
        self,
        script: str,
        timeout: float,
        on_output: Callable[[bytes], None] | None,
    ) -> tuple[int | None, bytes]:
        return self._e2b.run(script, timeout=timeout, on_output=on_output)

    def _destroy(self) -> None:
        """Kill the sandbox if this backend made it."""
        if self._owns_sandbox:
            self.sandbox.kill()

    @classmethod
    def connect(
        cls,
        sandbox_id: str,
        *,
        api_key: str | None = None,
        root: str = SANDBOX_ROOT,
        env: Mapping[str, str] | None = None,
        **options: Any,
    ) -> E2BBackend:
        """A backend over a sandbox that exists. It is never killed by the backend.

        Args:
            sandbox_id: The sandbox's id.
            api_key: The E2B API key; ``E2B_API_KEY`` in the environment
                when omitted.
            root: The workspace directory inside the sandbox.
            env: Variables added to every command's environment.
            options: Passed to the backend (``state_dir``, ``max_background``).
        """
        module = _e2b()
        sandbox = module.Sandbox.connect(sandbox_id, **_auth(api_key))
        return cls(sandbox, root=root, env=env, **options)

    @classmethod
    def create(
        cls,
        *,
        template: str | None = None,
        lifetime_s: int = 600,
        api_key: str | None = None,
        envs: Mapping[str, str] | None = None,
        root: str = SANDBOX_ROOT,
        env: Mapping[str, str] | None = None,
        **options: Any,
    ) -> E2BBackend:
        """Create a sandbox and return a backend that owns it: :meth:`close` kills it.

        Args:
            template: The sandbox template; the service default when omitted.
            lifetime_s: Seconds the sandbox lives before the service ends
                it, the backstop for one a crashed process leaves behind.
            api_key: The E2B API key; ``E2B_API_KEY`` when omitted.
            envs: Environment variables set on the sandbox itself.
            root: The workspace directory inside the sandbox.
            env: Variables added to every harness command's environment.
            options: Passed to the backend (``state_dir``, ``max_background``).
        """
        module = _e2b()
        sandbox = module.Sandbox.create(
            template=template,
            timeout=lifetime_s,
            envs=dict(envs) if envs else None,
            **_auth(api_key),
        )
        return cls(sandbox, root=root, env=env, owns_sandbox=True, **options)


def _auth(api_key: str | None) -> dict[str, Any]:
    return {"api_key": api_key} if api_key else {}


def _e2b() -> Any:
    """The ``e2b`` package; the only place it is imported."""
    try:
        import e2b
    except ImportError as exc:
        raise ImportError('E2BBackend needs the e2b package: pip install "locus-sdk[e2b]"') from exc
    return e2b
