import os
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from fastapi.testclient import TestClient

# Ensure environment has FACE_API_KEY
os.environ["FACE_API_KEY"] = "test-secret-key"

from app.main import app
from app.config import settings
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

    @patch("app.main.face_service.get_embedding")
    def test_enroll_oversized_upload_returns_413(self, mock_get_embedding):
        oversized_data = b"x" * (settings.max_image_size_bytes + 1)
        response = self.client.post(
            "/enroll",
            headers={"X-API-Key": "test-secret-key"},
            data={"name": "Bob"},
            files={"file": ("oversized.jpg", oversized_data, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("detail", response.json())
        mock_get_embedding.assert_not_called()

    @patch("app.main.face_service.get_embedding")
    def test_enroll_file_at_limit_accepted(self, mock_get_embedding):
        mock_emb = np.random.randn(512).astype(np.float32)
        mock_get_embedding.return_value = (mock_emb, (10, 10, 100, 100), 0.99)

        limit_data = b"x" * settings.max_image_size_bytes
        response = self.client.post(
            "/enroll",
            headers={"X-API-Key": "test-secret-key"},
            data={"name": "Bob"},
            files={"file": ("valid.jpg", limit_data, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        mock_get_embedding.assert_called_once()
        data = response.json()
        self.assertIn("enrolled successfully", data["message"])
        self.assertEqual(data["box"], [10, 10, 100, 100])
        self.assertEqual(data["det_conf"], 0.99)

        user = self.db.query(User).filter(User.name == "Bob").first()
        self.assertIsNotNone(user)

    @patch("app.main.face_service.get_embedding")
    def test_recognize_oversized_upload_returns_413(self, mock_get_embedding):
        oversized_data = b"x" * (settings.max_image_size_bytes + 1)
        response = self.client.post(
            "/recognize",
            headers={"X-API-Key": "test-secret-key"},
            files={"file": ("oversized.jpg", oversized_data, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("detail", response.json())
        mock_get_embedding.assert_not_called()

    @patch("app.main.face_service.get_embedding")
    def test_recognize_file_at_limit_accepted(self, mock_get_embedding):
        mock_emb = np.random.randn(512).astype(np.float32)
        mock_emb = mock_emb / np.linalg.norm(mock_emb)
        u = User(name="Charlie", embedding=mock_emb)
        self.db.add(u)
        self.db.commit()

        mock_get_embedding.return_value = (mock_emb, (15, 15, 120, 120), 0.95)

        limit_data = b"x" * settings.max_image_size_bytes
        response = self.client.post(
            "/recognize",
            headers={"X-API-Key": "test-secret-key"},
            files={"file": ("valid.jpg", limit_data, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200)
        mock_get_embedding.assert_called_once()
        data = response.json()
        self.assertEqual(data["match"], "Charlie")
        self.assertAlmostEqual(data["score"], 1.0, places=4)
        self.assertEqual(data["box"], [15, 15, 120, 120])


if __name__ == "__main__":
    unittest.main()
