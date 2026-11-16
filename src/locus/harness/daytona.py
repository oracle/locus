# Copyright (c) 2025, 2026 Oracle and/or its affiliates.
# Licensed under the Universal Permissive License v1.0 as shown at
# https://oss.oracle.com/licenses/upl/

"""A Daytona sandbox as a workspace.

`Daytona <https://www.daytona.io>`_ runs sandboxes as a hosted service; its
Python client (``pip install "locus-sdk[daytona]"``, the ``daytona``
package) runs a command with ``sandbox.process.exec`` and moves a file with
``sandbox.fs.upload_file`` / ``sandbox.fs.download_file``, which is
:class:`~locus.harness.session.SessionLike`. :class:`DaytonaSession` is the
adapter and everything else is
:class:`~locus.harness.session.SessionBackend`.

The client returns a command's exit code and its output as one string, and
reports a command that runs past ``timeout`` as a timeout error, which the
adapter turns back into a missing status. A sandbox can be stopped and
started by the service, so :meth:`DaytonaBackend.stop` and
:meth:`~DaytonaBackend.start` do that; the filesystem survives a stop.

This backend is written against the client's documented surface and
checked with a stand-in in the unit tests; the live test under
``tests/integration`` runs only when ``DAYTONA_API_KEY`` is set. It has not
been run against a Daytona account by the maintainers.

The ``daytona`` package is imported only by :meth:`DaytonaBackend.connect`
and :meth:`DaytonaBackend.create`. Pass a sandbox of your own to the
constructor and nothing from it is imported.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from locus.harness.local import MAX_BACKGROUND
from locus.harness.session import SessionBackend


__all__ = ["DaytonaBackend", "DaytonaSession"]


def _is(exc: BaseException, *names: str) -> bool:
    kind = type(exc).__name__
    return any(name in kind for name in names)


class DaytonaSession:
    """One Daytona sandbox as a :class:`~locus.harness.session.SessionLike`.

    Args:
        sandbox: A ``daytona.Sandbox``, or anything with its ``process.exec``,
            ``fs.upload_file`` and ``fs.download_file``.
    """

    def __init__(self, sandbox: Any) -> None:
        self.sandbox = sandbox

    def run(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
        """Run ``command`` in the sandbox's shell; its status and its output.

        The client returns output as one string once the command ends, so
        nothing streams; the backend hands it all to ``on_output`` at the end.
        """
        try:
            response = self.sandbox.process.exec(command, timeout=max(1, math.ceil(timeout)))
        except Exception as exc:
            if _is(exc, "Timeout"):
                return None, b""
            raise
        code = getattr(response, "exit_code", None)
        output = getattr(response, "result", None) or ""
        return (code if isinstance(code, int) else None), str(output).encode()

    def exec(self, command: str, *, timeout: float) -> tuple[int | None, bytes]:
        return self.run(command, timeout=timeout)

    def upload_file(self, path: str, data: bytes) -> None:
        self.sandbox.fs.upload_file(data, path)

    def download_file(self, path: str) -> bytes:
        try:
            data = self.sandbox.fs.download_file(path)
        except Exception as exc:
            if _is(exc, "NotFound"):
                raise FileNotFoundError(path) from exc
            raise
        if data is None:
            raise FileNotFoundError(path)
        return bytes(data)


class DaytonaBackend(SessionBackend):
    """A Daytona sandbox as a workspace. Isolated.

    Args:
        sandbox: A ``daytona.Sandbox`` or anything with the methods
            :class:`DaytonaSession` names, plus ``start``, ``stop`` and
            ``delete`` for the lifecycle.
        root: The workspace directory inside the sandbox; the sandbox's
            working directory (``get_work_dir()``) when omitted.
        owns_sandbox: Delete the sandbox on :meth:`close`. Set by
            :meth:`create`.
        env: Variables added to every command's environment.
        max_background: How many background commands may run at once.
        state_dir: Where background commands keep their state.
    """

    def __init__(
        self,
        sandbox: Any,
        *,
        root: str | None = None,
        owns_sandbox: bool = False,
        env: Mapping[str, str] | None = None,
        max_background: int = MAX_BACKGROUND,
        state_dir: str = "/tmp/.locus-harness",  # noqa: S108 - inside the sandbox
    ) -> None:
        self._daytona = DaytonaSession(sandbox)
        if root is None:
            root = str(sandbox.get_work_dir())
        sandbox_id = getattr(sandbox, "id", None) or "sandbox"
        super().__init__(
            self._daytona,
            root=root,
            label=f"daytona:{sandbox_id}",
            isolated=True,
            state_dir=state_dir,
            env=env,
            max_background=max_background,
        )
        self._owns_sandbox = owns_sandbox

    @property
    def sandbox(self) -> Any:
        """The sandbox every call goes through."""
        return self._daytona.sandbox

    def _start(self) -> None:
        """Start the sandbox again after a stop; its files are still there."""
        self.sandbox.start()

    def _stop(self) -> None:
        """End the background commands and stop the sandbox."""
        super()._stop()
        self.sandbox.stop()

    def _destroy(self) -> None:
        """Delete the sandbox if this backend made it."""
        if self._owns_sandbox:
            self.sandbox.delete()

    @classmethod
    def connect(
        cls,
        sandbox_id: str,
        *,
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        root: str | None = None,
        env: Mapping[str, str] | None = None,
        **options: Any,
    ) -> DaytonaBackend:
        """A backend over a sandbox that exists. It is never deleted by the backend.

        Args:
            sandbox_id: The sandbox's id or name.
            api_key: The Daytona API key; ``DAYTONA_API_KEY`` when omitted.
            api_url: The API URL; ``DAYTONA_API_URL`` when omitted.
            target: The target region; ``DAYTONA_TARGET`` when omitted.
            root: The workspace directory; the sandbox's working directory
                when omitted.
            env: Variables added to every command's environment.
            options: Passed to the backend (``state_dir``, ``max_background``).
        """
        client = _client(api_key, api_url, target)
        return cls(client.get(sandbox_id), root=root, env=env, **options)

    @classmethod
    def create(
        cls,
        *,
        snapshot: str | None = None,
        image: str | None = None,
        language: str | None = None,
        lifetime_minutes: int | None = 60,
        api_key: str | None = None,
        api_url: str | None = None,
        target: str | None = None,
        root: str | None = None,
        env: Mapping[str, str] | None = None,
        **options: Any,
    ) -> DaytonaBackend:
        """Create a sandbox and return a backend that owns it: :meth:`close` deletes it.

        Args:
            snapshot: A Daytona snapshot to start from; the default when
                neither it nor ``image`` is given.
            image: A container image to start from instead of a snapshot.
            language: The sandbox's language runtime, for a snapshot.
            lifetime_minutes: Minutes before the service deletes the sandbox
                on its own, the backstop for one a crashed process leaves
                behind; ``None`` for no backstop.
            api_key: The Daytona API key; ``DAYTONA_API_KEY`` when omitted.
            api_url: The API URL; ``DAYTONA_API_URL`` when omitted.
            target: The target region; ``DAYTONA_TARGET`` when omitted.
            root: The workspace directory; the sandbox's working directory
                when omitted.
            env: Variables added to every harness command's environment.
            options: Passed to the backend (``state_dir``, ``max_background``).
        """
        module = _daytona()
        client = _client(api_key, api_url, target)
        fields: dict[str, Any] = {}
        if language:
            fields["language"] = language
        if lifetime_minutes is not None:
            fields["ttl_minutes"] = lifetime_minutes
        if image:
            params = module.CreateSandboxFromImageParams(image=image, **fields)
        else:
            params = module.CreateSandboxFromSnapshotParams(snapshot=snapshot, **fields)
        sandbox = client.create(params)
        return cls(sandbox, root=root, env=env, owns_sandbox=True, **options)


def _client(api_key: str | None, api_url: str | None, target: str | None) -> Any:
    module = _daytona()
    fields = {
        k: v for k, v in (("api_key", api_key), ("api_url", api_url), ("target", target)) if v
    }
    return module.Daytona(module.DaytonaConfig(**fields)) if fields else module.Daytona()


def _daytona() -> Any:
    """The ``daytona`` package; the only place it is imported."""
    try:
        import daytona
    except ImportError as exc:
        raise ImportError(
            'DaytonaBackend needs the daytona package: pip install "locus-sdk[daytona]"'
        ) from exc
    return daytona
