import numpy as np
import numpy_financial as npf


class TCOCalculator:
    """
    Class used all over the project to calculate the consequence based on the methods provided in "Partnerskab for
    offentlige grønne indkøb"s tool "miljoestyrelsen-tco-vaerktoej-motorkoeretoejer". Made to be independent of
    xlwings dependencies unavailable to linux. The methods have been written directly from the tool and thus makes the
    same assumptions. Originals can be seen in src.fleetmanager.model.pogi
    """
    def __init__(self, **kwargs):
        """
        Parameters should be loaded with the initialisation.
        Most importantly is to define; "drivmiddel", "bil_type", "koerselsforbrug" (based on the actual allocated trips),

        Essential lambda functions here are:
            aarlig_braendstofforbrug    :   calculates the fuel usage - fossile: km / kml,
                                                                    electrical: km * wh / 1000
            aarlig_driftsomkostning :   calculates the yearly fule expense: usage * price * count
            nutidsvaerdi_drift  :   calculates the projected fuel expense:
                                    yearly fuel expense * (1 + interest rate / 100) ** -year_index

        Parameters
        ----------
        kwargs
        """

        # oplysninger om produktet
        self.etableringsgebyr = 0
        self.braendstofforbrug = 15
        self.leasingydelse = 0
        self.leasingtype = "operationel"
        self.ejerafgift = 0
        self.elforbrug = 200
        self.service = 0

        # oplysninger om brugeren
        self.antal = 1
        self.koerselsforbrug = 30000

        # baggrundsdata
        self.diskonteringsrente = 4
        self.evalueringsperiode = 4
        # 2020 = 0
        self.fremskrivnings_aar = 0
        self.prisstigning_benzin = 1.39
        self.prisstigning_diesel = 1.45
        self.pris_el = 2.13
        self.prisstigning_el = 1.67
        self.pris_benzin = 12.33
        self.pris_diesel = 10.83
        self.drivmiddel = "benzin"
        self.bil_type = "benzin"
        self.forsikring = 0
        self.loebende_omkostninger = 0
        self.foerste_aars_brugsperiode = 2021
        self.vaerdisaetning_tons_co2 = 1500

        # Fra CO2e udledninger Fremskrivningsarket
        self.co2e_udledninger_diesel = np.full(30, 2.98)
        self.co2e_udledninger_benzin = np.full(30, 2.52)
        self.co2e_udledninger_el = [
            0.089,
            0.07,
            0.058,
            0.054,
            0.05,
            0.042,
            0.037,
            0.032,
            0.013,
        ] + ([0.012] * 21)

        self.__dict__.update(kwargs)

        # functions
        self.fremskrivning = {
            "benzin": self.drivmiddel_udvikling(
                self.pris_benzin, self.prisstigning_benzin
            ),
            "diesel": self.drivmiddel_udvikling(
                self.pris_diesel, self.prisstigning_diesel
            ),
            "el": self.drivmiddel_udvikling(self.pris_el, self.prisstigning_el),
        }
        self.aarlig_braendstofforbrug = (
            lambda koersel, kml, drivmiddel: koersel / kml
            if drivmiddel.lower() != "el"
            else koersel * kml / 1000
        )  # kwh to wh
        self.aarlig_driftsomkostning = (
            lambda forbrug, pris, antal=1: forbrug * pris * antal
        )
        self.nutidsvaerdi_drift = (
            lambda driftsomkostning, rente, aar_index: driftsomkostning
            * (1 + rente / 100) ** -aar_index
        )

        # calculated
        self.driftsomkostninger_aar = self.driftsomkostninger()
        self.driftsomkostning = sum(self.driftsomkostninger_aar)
        self.omkostning = self.omkostninger()
        self.tco = self.driftsomkostning + self.omkostning + self.etableringsgebyr
        self.tco_average = self.tco_yearly()
        self.omkostning_average = self.omkostning_yearly()

    def driftsomkostninger(self):
        """
        Calculates the summed fuel expense on the vehicle.
        Returns
        -------
        list of expense on fuel over the selected evaluation period
        """
        if self.drivmiddel not in ["benzin", "diesel", "el"]:
            return [0]
        return [
            self.nutidsvaerdi_drift(
                self.aarlig_driftsomkostning(
                    self.aarlig_braendstofforbrug(
                        self.koerselsforbrug,
                        self.elforbrug
                        if self.drivmiddel.lower() == "el"
                        else self.braendstofforbrug,
                        self.drivmiddel,
                    ),
                    self.fremskrivning[self.drivmiddel][k - 1],
                ),
                self.diskonteringsrente,
                k,
            )
            for k in range(1, self.evalueringsperiode + 1)
        ]

    def drivmiddel_udvikling(self, pris, stigning):
        """
        Projecting the development in price of the fuel.

        Parameters
        ----------
        pris    :   int, current price
        stigning    :   int, percentage rate of fuel increase

        Returns
        -------
        list of fuel price for the next 30 years
        """
        udvikling = [pris]
        for _ in range(30):
            udvikling.append(udvikling[-1] * (1 + stigning / 100))
        return udvikling

    def omkostninger(self):
        """
        Summing the expenses not related to fuel expense over the evaluation period
        """
        return sum(
            (
                max(self.leasingydelse, 0)
                + max(self.ejerafgift, 0)
                + max(self.forsikring, 0)
                + max(self.service, 0)
                + max(self.loebende_omkostninger, 0)
            )
            * (1 + self.diskonteringsrente / 100) ** -aar
            for aar in range(1, self.evalueringsperiode + 1)
        )

    def omkostning_yearly(self):
        """
        Getting the yearly expense with the defined discount interest rate
        """
        return abs(
            npf.pmt(
                pv=self.omkostning,
                fv=0,
                rate=self.diskonteringsrente / 100,
                nper=self.evalueringsperiode,
            )
        )

    def tco_yearly(self):
        return abs(
            npf.pmt(
                pv=self.tco,
                fv=0,
                rate=self.diskonteringsrente / 100,
                nper=self.evalueringsperiode,
            )
        )

    def ekstern_miljoevirkning(self, sum_it=False):
        udledninger = []
        aarligt_forbrug_benzin_diesel = (
            0
            if self.braendstofforbrug == 0
            else self.koerselsforbrug / self.braendstofforbrug
        )
        el_aarligt_stroemforbrug = self.elforbrug / 1000 * self.koerselsforbrug
        el_hybrid_aarligt_forbrug = (
            0 if self.elforbrug == 0 else self.koerselsforbrug / self.elforbrug
        )
        if self.bil_type == "benzin":
            udledninger = [
                aarligt_forbrug_benzin_diesel * self.antal * k / 1000
                for k in self.co2e_udledninger_benzin
            ]
        elif self.bil_type == "diesel":
            udledninger = [
                aarligt_forbrug_benzin_diesel * self.antal * k / 1000
                for k in self.co2e_udledninger_diesel
            ]
        elif self.bil_type == "el":
            udledninger = [
                self.antal * el_aarligt_stroemforbrug * k / 1000
                for k in self.co2e_udledninger_el
            ]
        elif self.bil_type == "plugin hybrid benzin":
            udledninger_el = [
                self.antal * el_aarligt_stroemforbrug * k / 1000
                for k in self.co2e_udledninger_el
            ]
            udledninger_benzin = [
                el_hybrid_aarligt_forbrug * self.antal * k / 1000
                for k in self.co2e_udledninger_benzin
            ]
            udledninger = [a + b for a, b in zip(udledninger_el, udledninger_benzin)]
        elif self.bil_type == "plugin hybrid diesel":
            udledninger_el = [
                self.antal * el_aarligt_stroemforbrug * k / 1000
                for k in self.co2e_udledninger_el
            ]
            udledninger_diesel = [
                el_hybrid_aarligt_forbrug * self.antal * k / 1000
                for k in self.co2e_udledninger_diesel
            ]
            udledninger = [a + b for a, b in zip(udledninger_el, udledninger_diesel)]
        udledninger = udledninger[
            self.fremskrivnings_aar : self.fremskrivnings_aar + self.evalueringsperiode
        ]
        ekstern_virkninger = [
            udl
            * self.vaerdisaetning_tons_co2
            * (1 + self.diskonteringsrente / 100) ** -(k + 1)
            for k, udl in enumerate(udledninger)
        ]
        if sum_it:
            return sum(udledninger), sum(ekstern_virkninger)
        return udledninger, ekstern_virkninger
