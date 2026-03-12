import unittest
from unittest.mock import patch
from src.data.check_structure import check_existing_file

class TestCheckExistingFile(unittest.TestCase):

    @patch('builtins.input', return_value='y')
    def test_file_exists_overwrite(self, mock_input):
        self.assertTrue(check_existing_file('existing_file.txt'))

    @patch('builtins.input', return_value='n')
    def test_file_exists_do_not_overwrite(self, mock_input):
        self.assertFalse(check_existing_file('existing_file.txt'))

    def test_file_does_not_exist(self):
        self.assertTrue(check_existing_file('non_existent_file.txt'))

if __name__ == '__main__':
    unittest.main()
