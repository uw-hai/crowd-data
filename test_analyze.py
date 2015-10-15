import unittest
import analyze

class LoadDataTest(unittest.TestCase):
    def setUp(self):
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

    def test_combine(self):
        data_all = analyze.Data.from_rajpal_icml15(worker_type=None)
        data_ordinary = analyze.Data.from_rajpal_icml15(worker_type='ordinary')
        data_normal = analyze.Data.from_rajpal_icml15(worker_type='normal')
        data_master = analyze.Data.from_rajpal_icml15(worker_type='master')

        self.assertEqual(
            data_all.get_n_workers(),
            data_ordinary.get_n_workers() + data_normal.get_n_workers() + \
            data_master.get_n_workers())

    def test_columns(self):
        for df in (d.df for d in self.data.itervalues()):
            self.assertIn('worker', df)
            self.assertIn('question', df)
            self.assertIn('answer', df)
            self.assertIn('gt', df)
            self.assertIn('correct', df)

if __name__ == '__main__':
    unittest.main()
