import pytest
from coreason_identity.models import UserContext
from pydantic import EmailStr

@pytest.fixture
def mock_user_context():
    """Provides a valid UserContext for testing, matching coreason-identity 0.1.0 schema."""
    return UserContext(
        sub="test-user-001",
        email="test@example.com",
        permissions=["read", "write"]
    )
