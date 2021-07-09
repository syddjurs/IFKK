import sys
import unittest

import model


class TestDataClasses(unittest.TestCase):
    """
    Test VehicleModel dataclass
    """

    # vm1 = model.VehicleModel(name='vehicle', co2emission_pr_km=9.9)
    # vm2 = model.VehicleModel(name='vehicle', co2emission_pr_km=9.9)
    # vm3 = model.VehicleModel(name='vehicle2', co2emission_pr_km=9.9)
    # vm4 = model.VehicleModel(name='vehicle2', co2emission_pr_km=1.1)

    def _test_equal(self):
        self.assertEqual(self.vm1, self.vm2)

    def _test_not_equal(self):
        self.assertNotEqual(self.vm1, self.vm3)
        self.assertNotEqual(self.vm1, self.vm4)
        self.assertNotEqual(self.vm3, self.vm4)


class TestPogi(unittest.TestCase):
    """Test computation of POGI via Excel engine"""

    def test_operational_benzin(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "operational",
            "Benzin",
            etableringsgebyr=100000,
            braendstofforbrug=20,
            leasingydelse=25000,
            ejerafgift=500,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_op_benzin.xlsm",
        )

        self.assertAlmostEqual(tco, 162675, delta=0.5)
        self.assertAlmostEqual(eks, 7.56, delta=0.01)

    def test_operational_diesel(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "operational",
            "Diesel",
            etableringsgebyr=100000,
            braendstofforbrug=20,
            leasingydelse=25000,
            ejerafgift=500,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_op_diesel.xlsm",
        )

        self.assertAlmostEqual(tco, 162369, delta=0.5)
        self.assertAlmostEqual(eks, 8.94, delta=0.01)

    def test_operational_el(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "operational",
            "El",
            etableringsgebyr=100000,
            elforbrug=150,
            braendstofforbrug=20,
            leasingydelse=25000,
            ejerafgift=500,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_op_el.xlsm",
        )

        self.assertAlmostEqual(tco, 157650, delta=0.5)
        self.assertAlmostEqual(eks, 0.61, delta=0.01)

    def test_operational_pluginhybridbenzin(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "operational",
            "Plugin hybrid benzin",
            etableringsgebyr=100000,
            elforbrug=150,
            braendstofforbrug=20,
            leasingydelse=25000,
            ejerafgift=500,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_op_pluginbenzin.xlsm",
        )

        self.assertAlmostEqual(tco, 164643, delta=0.5)
        self.assertAlmostEqual(eks, 1.62, delta=0.01)

    def test_operational_pluginhybriddiesel(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "operational",
            "Plugin hybrid diesel",
            etableringsgebyr=100000,
            elforbrug=150,
            braendstofforbrug=20,
            leasingydelse=25000,
            ejerafgift=500,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_op_plugindiesel.xlsm",
        )

        self.assertAlmostEqual(tco, 163888, delta=0.5)
        self.assertAlmostEqual(eks, 1.80, delta=0.01)

    def test_financial_benzin(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "financial",
            "Benzin",
            indkobspris=100000,
            braendstofforbrug=20,
            serviceaftale=500,
            leasingydelse=25000,
            ejerafgift=500,
            tilbagetagningspris=50000,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_fin_benzin.xlsm",
        )

        self.assertAlmostEqual(tco, 152813, delta=0.5)
        self.assertAlmostEqual(eks, 7.56, delta=0.001)

    def test_financial_diesel(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "financial",
            "Diesel",
            indkobspris=100000,
            braendstofforbrug=20,
            serviceaftale=500,
            leasingydelse=25000,
            ejerafgift=500,
            tilbagetagningspris=50000,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_fin_diesel.xlsm",
        )

        self.assertAlmostEqual(tco, 152507, delta=0.5)
        self.assertAlmostEqual(eks, 8.94, delta=0.01)

    def test_financial_el(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "financial",
            "El",
            indkobspris=100000,
            braendstofforbrug=20,
            elforbrug=150,
            serviceaftale=500,
            leasingydelse=25000,
            ejerafgift=500,
            tilbagetagningspris=50000,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_fin_diesel.xlsm",
        )

        self.assertAlmostEqual(tco, 147787, delta=0.5)
        self.assertAlmostEqual(eks, 0.61, delta=0.01)

    def test_financial_pluginhybridbenzin(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "financial",
            "Plugin hybrid benzin",
            indkobspris=100000,
            braendstofforbrug=20,
            elforbrug=150,
            serviceaftale=500,
            leasingydelse=25000,
            ejerafgift=500,
            tilbagetagningspris=50000,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_fin_Plugin_hybrid_benzin.xlsm",
        )

        self.assertAlmostEqual(tco, 154780, delta=0.5)
        self.assertAlmostEqual(eks, 1.62, delta=0.01)

    def test_financial_pluginhybriddiesel(self):
        m = model.PogiExcel()

        tco, eks = m.compute(
            "financial",
            "Plugin hybrid diesel",
            indkobspris=100000,
            braendstofforbrug=20,
            elforbrug=150,
            serviceaftale=500,
            leasingydelse=25000,
            ejerafgift=500,
            tilbagetagningspris=50000,
            koerselsforbrug=15000,
            forsikring=5000,
            loebende_omkostninger=1000,
            save_filename=r"C:\Users\AllanLyckegaard\Downloads\tco_test_fin_Plugin_hybrid_diesel.xlsm",
        )

        self.assertAlmostEqual(tco, 154026, delta=0.5)
        self.assertAlmostEqual(eks, 1.80, delta=0.01)


if __name__ == "__main__":
    # suite = unittest.TestLoader().loadTestsFromModule( sys.modules[__name__] )
    # print(suite)
    # unittest.TextTestRunner(verbosity=3).run( suite )
    unittest.main(verbosity=3)
