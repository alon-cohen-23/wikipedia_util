from ner.line_info import LineInfo

import unittest


class TestLineInfo(unittest.TestCase):
    TEST_LINE = r"'word1  word2- \word3 word4"

    def test_ctor(self):
        line_info = LineInfo(TestLineInfo.TEST_LINE)
        self.assertEqual(line_info.words, ["'word1", 'word2-', '\\word3', 'word4'])

    def test_get_name(self):
        line_info = LineInfo(TestLineInfo.TEST_LINE)
        self.assertEqual(line_info.get_name(0, 1), "'word1")
        self.assertEqual(line_info.get_name(0, 2), "'word1  word2-")

    def test_strip(self):
        line_info = LineInfo(TestLineInfo.TEST_LINE)
        print(line_info.word_spans)

        line_info.strip_punctuation(0, 2)
        self.assertEqual(line_info.get_name(0, 1), "word1")
        self.assertEqual(line_info.get_name(0, 2), "word1  word2")

        line_info.strip_punctuation(1, 3)
        self.assertEqual(line_info.get_name(1, 3), "word2- \\word3")


if __name__ == '__main__':
    unittest.main()
