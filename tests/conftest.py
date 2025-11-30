"""
Pytest configuration for tests
"""

from dotenv import load_dotenv


def pytest_configure(config):
    """Load environment variables before running tests"""
    load_dotenv()
