import unittest
from montecarlo import pmt, MC

class TestPMT(unittest.TestCase):
    
    def test_pmt123456(self):
        pmts = [pmt(i+1) for i in range(6)]
    
    def test_bounds(self):
        with self.assertRaises(ValueError):
            pmt(0)
        with self.assertRaises(ValueError):
            pmt(7)

class TestMCCount(unittest.TestCase):
        
    def test(self):
        self.mc = MC(pmt(3), pmt(4), pmt(5))
        self.mc.random_ray(N=1000)
        self.mc.run()
        
        print("count_345 = {} / {}".format(self.mc.count(), self.mc.number_of_rays))
    
        self.assertEqual(self.mc.count([True])[0].n, self.mc.count(True).n)

class TestMCGeom(unittest.TestCase):
    
    def test(self):
        self.mc = MC(pmt(3), pmt(4))
        self.mc.random_ray(N=1000)
        self.mc.sample_geometry(100)
        self.mc.run(randgeom=True)

        count = self.mc.count()
        self.assertEqual(count.shape, (100,))
    
        count = self.mc.count([1,1], [1,0])
        self.assertEqual(count.shape, (2, 100))

if __name__ == '__main__':
    unittest.main()
