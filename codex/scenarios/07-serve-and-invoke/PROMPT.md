Serve a Locus agent over HTTP and test it offline. Install `locus-sdk` with the `server`
extra as well (`"locus-sdk[oci,server]"`, or the same extras on the install spec you were
given) and `httpx`.

1. Write `server.py`: an `Agent` on `ScriptedModel(["The order was delivered."], repeat_last=True)`,
   wrapped in `AgentServer(agent=..., api_key="test-key")`, with `app = server.app` at module
   scope.
2. Write `test_server.py` using `fastapi.testclient.TestClient(app)`: `GET /health` is 200;
   `POST /invoke` with `Authorization: Bearer test-key` and `{"prompt": "..."}` returns 200
   and a `message` containing `delivered`; the same call without the header is refused (401 or
   403).
3. Run the tests; all must pass. Do not start a network server.
