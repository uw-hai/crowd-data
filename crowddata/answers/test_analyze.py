"""Test cases for analyze.py."""
import unittest
from . import analyze


class LoadDataTest(unittest.TestCase):
    """Class to test loading live datasets.

    Only works if following environment variables are set:
        BRAGG_TEACH_DIR, BRAGG_HCOMP13_DIR, LIN_AAAI12_DIR, RAJPAL_ICML15_DIR

    """

    def setUp(self):
        """Setup."""
        self.data = dict()
        self.data['lin_tag'] = analyze.Data.from_lin_aaai12(workflow='tag')
        self.data['lin_wiki'] = analyze.Data.from_lin_aaai12(workflow='wiki')
        self.data['bragg'] = analyze.Data.from_bragg_hcomp13(
            positive_only=False)
        self.data['bragg_pos'] = analyze.Data.from_bragg_hcomp13(
            positive_only=True)
        self.data['rajpal_all'] = analyze.Data.from_rajpal_icml15(
            worker_type=None)
        self.data['rajpal_ordinary'] = analyze.Data.from_rajpal_icml15(
            worker_type='ordinary')
        self.data['rajpal_normal'] = analyze.Data.from_rajpal_icml15(
            worker_type='normal')
        self.data['rajpal_master'] = analyze.Data.from_rajpal_icml15(
            worker_type='master')
        self.data['bragg_teach'] = analyze.Data.from_bragg_teach()

    def test_rajpal(self):
        """Test slices of Rajpal data load correctly."""
        self.assertEqual(
            self.data['rajpal_all'].get_n_workers(),
            self.data['rajpal_ordinary'].get_n_workers() +
            self.data['rajpal_normal'].get_n_workers() +
            self.data['rajpal_master'].get_n_workers())

    def test_columns(self):
        """Test data has required columns."""
        for df in (d.df for d in self.data.itervalues()):
            self.assertIn('worker', df)
            self.assertIn('question', df)
            self.assertIn('answer', df)
            self.assertIn('gt', df)
            self.assertIn('correct', df)

#    def test_bragg_teach(self):
#        """Test BraggTeach data loads correctly.
#
#        Assumes data contains all conditions.
#
#        """
#        self.data['bragg_teach'].df

if __name__ == '__main__':
    unittest.main()
