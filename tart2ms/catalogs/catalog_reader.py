import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from pandas import read_csv
import os

class catalog_source:
    def __init__(self, name, rahms, decdms, flux, spi, reffreq=1.5e9, epoch="J2000", frame="fk5"):
        """
            name: source name
            ra: ra in hms string
            dec: dec in dms string
            flux: intrisic source flux in Jy at ref freq
            spi: source spectral index
            reffreq: reference frequency in Hz
            frame: see astropy SkyCoord
            epoch: see astropy SkyCoord equinox options
        """
        self.__name = name
        self.__skypos = SkyCoord(rahms + " " + decdms, 
                                 equinox=epoch, frame=frame)
        self.__flux = flux
        self.__spi = spi
        self.__reffreq = reffreq
        self.__frame = frame
        self.__epoch = epoch

    @property
    def name(self):
        return self.__name

    @property
    def ra(self):
        """ Right ascention in ICRS frame """
        return self.__skypos.icrs.ra.hms

    @property
    def dec(self):
        """ Declination in ICRS frame """
        return self.__skypos.icrs.dec.dms

    @property
    def radeg(self):
        """ Right ascention in ICRS frame """
        return self.__skypos.icrs.ra.value

    @property
    def decdeg(self):
        """ Declination in ICRS frame """
        return self.__skypos.icrs.dec.value

    @property
    def rarad(self):
        """ Right ascention in ICRS frame """
        return np.deg2rad(self.__skypos.icrs.ra.value)

    @property
    def decrad(self):
        """ Declination in ICRS frame """
        return np.deg2rad(self.__skypos.icrs.dec.value)

    def flux(self, freq):
        return (freq / self.__reffreq)**self.__spi * \
                self.__flux

class catalog_factory:
    @classmethod
    def from_3CRR(cls,
                  filename=os.path.join(os.path.split(os.path.abspath(__file__))[0], 
                                        "3CRR"),
                  fluxlim15=1.0):
        df = read_csv(filename, delimiter="\t")
        zipcols =["Name", "S 178-MHz", "Radio RA", "Radio DEC", "SPI"]
        if not set(zipcols).issubset(set(df.columns)):
            x = ",".join(zipcols)
            raise RuntimeError(f"Expected columns in 3CRR catalog {x}")
        collection = []
        for name, flux, ra, dec, spi in filter(lambda x: x[1] * (1.5e9 / 178e6)**-x[4] >= fluxlim15,
                                               zip(*map(lambda c: df[c].values, zipcols))):
            rah, ram, ras = tuple(map(lambda x: x.strip(), ra.strip().split(" ")))
            decd, decm, decs = tuple(map(lambda x: x.strip(), dec.strip().split(" ")))
            rahms = f"{rah}h{ram}m{ras}s"
            decdms = f"{decd}d{decm}m{decs}s"
            collection.append(catalog_source(name, rahms, decdms, flux, -spi, reffreq=178e6))
        
        return collection
    
    @classmethod
    def from_SUMMS(cls,
                   filename=os.path.join(os.path.split(os.path.abspath(__file__))[0], 
                                        "SUMMS"),
                   fluxlim15=0.1):
        df = read_csv(filename, delimiter=" ")
        zipcols =["RAh", "RAm", "RAs", "DECd", "DECm", "DECs", "IS"]
        if not set(zipcols).issubset(set(df.columns)):
            x = ",".join(zipcols)
            raise RuntimeError(f"Expected columns in SUMMS catalog {x}")
        collection = []
        for rah, ram, ras, decd, decm, decs, fluxmjy in  filter(lambda x: x[6] *1e-3 * (1.5e9 / 843e6)**-0.7 >= fluxlim15,
                                                                zip(*map(lambda c: df[c].values, zipcols))):
            name = f"J{rah}{ram}{decd}{decm}"
            rahms = f"{rah}h{ram}m{ras}s"
            decdms = f"{decd}d{decm}m{decs}s"
            collection.append(catalog_source(name, rahms, decdms, fluxmjy*1e-3, -0.7, reffreq=843e6))
        
        return collection

    @classmethod
    def from_MKGains(cls,
                     filename=os.path.join(os.path.split(os.path.abspath(__file__))[0], 
                                           "MeerKAT_gaincal_list"),
                     fluxlim15=1.0):
        df = read_csv(filename, delimiter="\t")
        zipcols = ["name", "S1.4GHz (Jy)", "RA (J2000)", "DEC (J2000)", "Alpha"]
        if not set(zipcols).issubset(set(df.columns)):
            x = ",".join(zipcols)
            raise RuntimeError(f"Expected columns in MeerKAT calibrator catalog {x}")
        df["S1.4GHz (Jy)"] = np.array(map(lambda x: float(x.split("+/-")[0].strip()), df["S1.4GHz (Jy)"]))
        df["Alpha"] = np.array(map(lambda x: float(x.split("+/-")[0].strip()), df["Alpha"]))
        collection = []
        for name, flux, ra, dec, spi in filter(lambda x: x[1] * (1.5e9 / 1.4e9)**x[4] >= fluxlim15,
                                               zip(*map(lambda c: df[c].values, zipcols))):
            rah, ram, ras = tuple(map(lambda x: x.strip(), ra.strip().split(":")))
            decd, decm, decs = tuple(map(lambda x: x.strip(), dec.strip().split(":")))
            rahms = f"{rah}h{ram}m{ras}s"
            decdms = f"{decd}d{decm}m{decs}s"
            collection.append(catalog_source(name, rahms, decdms, flux, -spi, reffreq=178e6))
        
        return collection