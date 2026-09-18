class WokeError(Exception):
    """Base error for woke."""


class UnknownKind(WokeError):
    def __init__(self, kind: str) -> None:
        super().__init__(f"unknown event kind: {kind}")
        self.kind = kind


class ValidationError(WokeError):
    pass


class NotFound(WokeError):
    pass


class SessionBusy(WokeError):
    pass


class PathEscapes(WokeError):
    def __init__(self, path: str) -> None:
        super().__init__(f"path escapes workspace: {path}")
        self.path = path


class HostDown(WokeError):
    pass


class HostLocked(WokeError):
    """Another woke process already holds this workspace state root."""


class AuthError(WokeError):
    pass


class ModelError(WokeError):
    """Provider or transport failure talking to the model."""


class SimulatedCrash(BaseException):
    """Test-only crash after a committed tool.call. Not a subclass of Exception."""
