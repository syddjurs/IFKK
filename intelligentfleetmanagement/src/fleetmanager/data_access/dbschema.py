from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class Trips(Base):
    __tablename__ = "trips"
    id = Column(Integer, primary_key=True)
    car_id = Column(Integer, ForeignKey("cars.id"), index=True)
    distance = Column(Float)
    start_time = Column(DateTime, index=True)
    end_time = Column(DateTime, index=True)
    start_latitude = Column(Float)
    start_longitude = Column(Float)
    end_latitude = Column(Float)
    end_longitude = Column(Float)
    driver_name = Column(String(128))
    department = Column(String(128))
    start_location = Column(Integer, ForeignKey("allowed_starts.id"))


class RoundTrips(Base):
    __tablename__ = "roundtrips"
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, index=True)
    end_time = Column(DateTime, index=True)
    start_latitude = Column(Float)
    start_longitude = Column(Float)
    end_latitude = Column(Float)
    end_longitude = Column(Float)
    distance = Column(Float)
    car_id = Column(Integer, ForeignKey("cars.id"), index=True)
    driver_name = Column(String(128))
    start_location_id = Column(Integer, ForeignKey("allowed_starts.id"))


class Cars(Base):
    __tablename__ = "cars"
    id = Column(Integer, primary_key=True)
    plate = Column(String(128))
    make = Column(String(128))
    model = Column(String(128))
    type = Column(Integer, ForeignKey("vehicle_types.id"))
    fuel = Column(Integer, ForeignKey("fuel_types.id"))
    wltp_fossil = Column(Float)
    wltp_el = Column(Float)
    capacity_decrease = Column(
        Float
    )  # percentage if range is less than expected (e.g. elbil 80% range during winter)
    co2_pr_km = Column(Float)
    range = Column(Float)
    omkostning_aar = Column(Float)
    location = Column(Integer, ForeignKey("allowed_starts.id"))
    start_leasing = Column(DateTime)
    end_leasing = Column(DateTime)
    leasing_type = Column(Integer, ForeignKey("leasing_types.id"))
    km_aar = Column(Float)  # hvis der findes km-forbrug på leasingaftalen
    sleep = Column(
        Integer
    )  # Amount of hours electric vehicles needs for charging each day


class AllowedStarts(Base):
    __tablename__ = "allowed_starts"
    id = Column(Integer, primary_key=True)
    address = Column(String(128))
    latitude = Column(Float)
    longitude = Column(Float)
    cars = relationship("Cars")


class LeasingTypes(Base):
    __tablename__ = "leasing_types"
    id = Column(Integer, primary_key=True)
    name = Column(String(128))
    cars = relationship("Cars")


class FuelTypes(Base):
    __tablename__ = "fuel_types"
    id = Column(Integer, primary_key=True)
    name = Column(String(128))
    refers_to = Column(Integer)
    cars = relationship("Cars")


class VehicleTypes(Base):
    __tablename__ = "vehicle_types"
    id = Column(Integer, primary_key=True)
    name = Column(String(128))
    refers_to = Column(Integer)
    cars = relationship("Cars")


default_leasing_types = [
    dict(id=1, name="operationel"),
    dict(id=2, name="finansiel"),
    dict(id=3, name="ejet"),
]

default_fuel_types = [
    dict(id=1, name="benzin", refers_to=1),
    dict(id=2, name="diesel", refers_to=2),
    dict(id=3, name="el", refers_to=3),
    dict(id=4, name="hybrid", refers_to=1),
    dict(id=5, name="plugin hybrid benzin", refers_to=1),
    dict(id=6, name="plugin hybrid diesel", refers_to=2),
    dict(id=7, name="electric3", refers_to=3),
    dict(id=8, name="electric", refers_to=3),
    dict(id=9, name="petrol", refers_to=1),
    dict(id=10, name="bike", refers_to=10),
]

default_vehicle_types = [
    dict(id=1, name="cykel", refers_to=1),
    dict(id=2, name="elcykel", refers_to=2),
    dict(id=3, name="elbil", refers_to=3),
    dict(id=4, name="fossilbil", refers_to=4),
]
