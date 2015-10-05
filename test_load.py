import unittest
import load
import analyze

class RajpalDataTest(unittest.TestCase):
    def test_worker_indices(self):
        data_all = analyze.Data(
            load.load_rajpal_icml15(worker_type=None))
        data_ordinary = analyze.Data(
            load.load_rajpal_icml15(worker_type='ordinary'))
        data_normal = analyze.Data(
            load.load_rajpal_icml15(worker_type='normal'))
        data_master = analyze.Data(
            load.load_rajpal_icml15(worker_type='master'))

        self.assertEqual(
            data_all.get_n_workers(),
            data_ordinary.get_n_workers() + data_normal.get_n_workers() + \
            data_master.get_n_workers())

if __name__ == '__main__':
    unittest.main()
