import os
import sys
import unittest

from unittest.mock import patch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)  # FIXME: this is a workaround

from utils import get_data_file_names, reset_chroma_db, CHROMA_PATH


class TestUtils(unittest.TestCase):
    @patch("os.listdir")
    def test_get_data_file_names_exists(self, mock_listdir):
        """Tests the return of a list of filenames in the data folder"""
        mock_listdir.return_value = ["document_1.pdf", "document_2.pdf"]
        result = get_data_file_names()

        self.assertEqual(result, ["document_1.pdf", "document_2.pdf"])

    @patch("os.listdir")
    def test_get_data_file_names_not_exists(self, mock_listdir):
        """Tests the return of an empty list if the data folder does not exist"""
        mock_listdir.side_effect = FileNotFoundError
        result = get_data_file_names()

        self.assertEqual(result, [])

    @patch("os.path.exists")
    @patch("shutil.rmtree")
    def test_reset_chroma_db(self, mock_rmtree, mock_exists):
        """Tests the removal and recreation of the Chroma folder"""
        mock_exists.return_value = True
        reset_chroma_db(CHROMA_PATH)
        mock_rmtree.assert_called_once_with(CHROMA_PATH)

        mock_exists.return_value = False
        reset_chroma_db(CHROMA_PATH)


if __name__ == "__main__":
    unittest.main()
