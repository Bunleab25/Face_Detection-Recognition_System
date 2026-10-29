import os
import unittest
from unittest.mock import patch

from fastapi import HTTPException

from app.security import require_api_key


class RequireApiKeyTests(unittest.TestCase):
    def test_missing_server_key_fails_closed(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(HTTPException) as error:
                require_api_key("supplied")

        self.assertEqual(error.exception.status_code, 503)

    def test_missing_or_incorrect_client_key_is_rejected(self):
        with patch.dict(os.environ, {"FACE_API_KEY": "expected"}, clear=True):
            for supplied_key in (None, "incorrect", "é"):
                with self.subTest(supplied_key=supplied_key):
                    with self.assertRaises(HTTPException) as error:
                        require_api_key(supplied_key)
                    self.assertEqual(error.exception.status_code, 401)

    def test_correct_key_is_accepted(self):
        with patch.dict(os.environ, {"FACE_API_KEY": "expected"}, clear=True):
            self.assertIsNone(require_api_key("expected"))


if __name__ == "__main__":
    unittest.main()
