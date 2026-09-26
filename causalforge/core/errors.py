"""Structured error taxonomy (C100-C500) for CausalForge."""


class CausalForgeError(Exception):
    """Base class for all CausalForge errors. Carries a stable code."""

    code = "C000"

    def __init__(self, message: str = ""):
        self.message = message
        super().__init__(f"[{self.code}] {message}")


class ConfigError(CausalForgeError):
    code = "C100"


class DataError(CausalForgeError):
    code = "C200"


class EstimatorError(CausalForgeError):
    code = "C300"


class FitError(EstimatorError):
    code = "C310"


class PredictError(EstimatorError):
    code = "C320"


class BackendUnavailableError(CausalForgeError):
    code = "C400"

    def __init__(self, backend: str = ""):
        self.backend = backend
        super().__init__(f"Backend '{backend}' is not available; offline fallback engaged.")


class PipelineError(CausalForgeError):
    code = "C500"
