from .compare import router as compare_router
from .generator import router as generator_router
from .optimization import router as optimization_router
from .steps import router as steps_router

__all__ = ["optimization_router", "steps_router", "generator_router", "compare_router"]
