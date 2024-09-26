import os
import sys
import unittest

from unittest.mock import patch

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(root_dir)  # FIXME: this is a workaround

from populate_db import load_documents, save_to_chroma


class TestPopulateDB(unittest.TestCase):
    @patch("populate_db.PyPDFDirectoryLoader.load")
    def test_load_documents(self, mock_load):
        mock_load.return_value = ["document_1.pdf", "document_2.pdf"]
        documents = load_documents()

        self.assertEqual(documents, ["document_1.pdf", "document_2.pdf"])

    @patch("populate_db.reset_chroma_db")
    @patch("populate_db.initialise_chroma")
    @patch("populate_db.initialise_embeddings")
    def test_save_to_chroma(self, mock_initialise_embeddings, mock_initialise_chroma, mock_reset_chroma_db):
        mock_initialise_embeddings.return_value = "mock_embeddings"
        mock_initialise_chroma.return_value.add_documents.return_value = None  # empty return

        save_to_chroma(["part_1", "part_2"])

        # calls functions once to simulate saving two parts
        mock_reset_chroma_db.assert_called_once()
        mock_initialise_embeddings.assert_called_once()
        mock_initialise_chroma.assert_called_once()

        mock_initialise_chroma.return_value.add_documents.assert_called_once_with(documents=["part_1", "part_2"])


if __name__ == "__main__":
    unittest.main()
