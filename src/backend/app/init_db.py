from .db import Base, engine
from . import models_db  # noqa: F401


def init_db() -> None:
    Base.metadata.create_all(bind=engine)

