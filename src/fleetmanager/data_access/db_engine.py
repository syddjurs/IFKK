import os
from contextlib import contextmanager

import pandas as pd
import sqlalchemy
from dotenv import load_dotenv
from importlib_resources import files
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from .dbschema import (
    Base,
    FuelTypes,
    LeasingTypes,
    VehicleTypes,
    default_fuel_types,
    default_leasing_types,
    default_vehicle_types,
)

try:
    import datasources
    from pgrace import properties
except ImportError:
    GRACE_DATA_SOURCE = ""
else:
    GRACE_DATA_SOURCE = properties.get_shared_property("DATA_SOURCE")

load_dotenv()


def engine_creator(
    db_name=None, db_password=None, db_user=None, db_url=None, db_server=None,
) -> sqlalchemy.engine.Engine:
    """
    Generic db engine creator. Loads env variables, e.g. in .env otherwise could be passed with click.
    Ensures that tables according to dbschema is created before returning

    Parameters
    ----------
    db_name
    db_password
    db_user
    db_url

    Returns
    -------
    sqlalchemy.engine
    """
    if db_name is None:
        db_name = os.getenv("DB_NAME")
    if db_password is None:
        db_password = os.getenv("DB_PASSWORD")
    if db_user is None:
        db_user = os.getenv("DB_USER")
    if db_url is None:
        db_url = os.getenv("DB_URL")
    if db_server is None:
        db_server = os.getenv("DB_SERVER")

    if any((db_name, db_password, db_user, db_url, db_server)):
        db_engine = create_engine(
            f"{db_server}://{db_user}:{db_password}@{db_url}/{db_name}",
            encoding="latin-1",
        )
    elif GRACE_DATA_SOURCE:
        db_engine = datasources.connect(GRACE_DATA_SOURCE)
    else:
        from sqlite3 import OperationalError

        db_engine = create_engine(
            "sqlite:///file:fleetdb?mode=memory&cache=shared&uri=true",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            encoding="latin-1",
        )
        try:
            db_engine.raw_connection().connection.executescript(
                files("fleetmanager")
                .joinpath("dummy_data.sql")
                .read_text(encoding="utf-8")
            )
        except OperationalError:
            pass

    Base.metadata.create_all(db_engine)
    create_defaults(db_engine)

    return db_engine


def create_defaults(engine_):
    """
    Function to load in the defaults defined in dbschema
    """
    forms = [VehicleTypes, LeasingTypes, FuelTypes]
    default_entries = [default_vehicle_types, default_leasing_types, default_fuel_types]
    for k, (form, defaults) in enumerate(zip(forms, default_entries)):
        adds = []
        current = pd.read_sql(select([form]), engine_)
        for default in defaults:
            if default["id"] not in current.id.values:
                adds.append(form(**default))
        if adds:
            Session = session_factory(engine_)
            with Session.begin() as sess:
                sess.add_all(adds)


def session_factory(db_engine):
    """
    Generating sessions used all over the project by feeding in a sqlalchemy engine.
    Handles rollback if a connection issue or the like occurs.

    Parameters
    ----------
    db_engine   :   sqlalchemy.engine, the database connection

    Returns
    -------
    ManagedSession
    """
    Session = sessionmaker(bind=db_engine)

    class ManagedSession:
        @contextmanager
        def __call__(self):
            session = Session()
            try:
                yield session
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        @contextmanager
        def begin(self):
            session = Session()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

    return ManagedSession()
