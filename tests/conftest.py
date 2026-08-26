import pytest

from src.spark_session import get_spark


@pytest.fixture(scope="session")
def spark():
    session = get_spark(app_name="pytest_session")
    yield session
    session.stop()
