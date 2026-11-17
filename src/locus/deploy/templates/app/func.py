"""OCI Functions entry: every invocation becomes a request on the AgentServer app."""

from locus.deploy.functions import asgi_handler


handler = asgi_handler("__LOCUS_APP__")
