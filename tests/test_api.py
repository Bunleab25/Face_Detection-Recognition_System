import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from fastapi.testclient import TestClient

# Ensure environment has FACE_API_KEY
os.environ["FACE_API_KEY"] = "test-secret-key"

from app.main import app
from app.database import get_db, Base, User, engine
from sqlalchemy.orm import sessionmaker

TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class ApiEndpointsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        Base.metadata.create_all(bind=engine)
        cls.client = TestClient(app)

    def setUp(self):
        self.db = TestSessionLocal()
        # Clean up users table for clean test
        self.db.query(User).delete()
        self.db.commit()

    def tearDown(self):
        self.db.query(User).delete()
        self.db.commit()
        self.db.close()

    def test_status_endpoint(self):
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "online")
        self.assertIn("device", data)
        self.assertIn("enrolled_count", data)

    def test_verify_key_unauthorized(self):
        response = self.client.get("/api/verify-key")
        self.assertEqual(response.status_code, 401)

        response = self.client.get("/api/verify-key", headers={"X-API-Key": "wrong"})
        self.assertEqual(response.status_code, 401)

    def test_verify_key_authorized(self):
        response = self.client.get(
            "/api/verify-key", headers={"X-API-Key": "test-secret-key"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["valid"])

    def test_list_and_delete_users(self):
        # Insert mock user
        mock_embedding = np.random.randn(512).astype(np.float32)
        u = User(name="Alice", embedding=mock_embedding)
        self.db.add(u)
        self.db.commit()
        self.db.refresh(u)
        user_id = u.id

        # List users
        res = self.client.get("/users", headers={"X-API-Key": "test-secret-key"})
        self.assertEqual(res.status_code, 200)
        users = res.json()
        self.assertEqual(len(users), 1)
        self.assertEqual(users[0]["name"], "Alice")
        self.assertEqual(users[0]["id"], user_id)

        # Delete user
        del_res = self.client.delete(
            f"/users/{user_id}", headers={"X-API-Key": "test-secret-key"}
        )
        self.assertEqual(del_res.status_code, 200)
        self.assertIn("Alice", del_res.json()["message"])

        # Verify list is empty
        res2 = self.client.get("/users", headers={"X-API-Key": "test-secret-key"})
        self.assertEqual(res2.json(), [])

    def test_delete_nonexistent_user(self):
        del_res = self.client.delete(
            "/users/99999", headers={"X-API-Key": "test-secret-key"}
        )
        self.assertEqual(del_res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
