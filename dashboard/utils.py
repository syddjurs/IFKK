import os
import time

import pandas as pd
from bokeh.models import ColumnDataSource, DataTable, TableColumn
from sqlalchemy import create_engine
from xdg import BaseDirectory

import vehicle

DAY = 86400


def data_from_db(conn_info):
    """TODO: Docstring for data_from_db.

    Parameters
    ----------
    conn_info : TODO

    Returns
    -------
    TODO

    """
    data_path = os.path.join(
        BaseDirectory.save_cache_path(
            os.path.join("fleetmanager", conn_info["server"])
        ),
        conn_info["schema"],
    )
    try:
        ctime = os.path.getctime(data_path)
    except FileNotFoundError:
        ctime = 0

    if ctime + DAY > time.time():
        dataset = pd.read_hdf(data_path, key="df")
    else:
        db_connection = create_engine(
            "mysql+pymysql://{}:{}@{}/{}".format(
                conn_info["user"],
                conn_info["pass"],
                conn_info["server"],
                conn_info["schema"],
            )
        )
        dataset = pd.read_sql(
            "SELECT "
            "start_time, "
            "end_time, "
            "car_id, "
            "start_latitude ,"
            "start_longitude ,"
            "distance, "
            "department "
            "FROM trips",
            con=db_connection,
        )
        dataset.department.replace([None], "Ukendt", inplace=True)
        dataset.to_hdf(data_path, key="df")

    dataset["current"] = [vehicle.Unassigned()] * dataset.shape[0]
    dataset["simulation"] = [vehicle.Unassigned()] * dataset.shape[0]

    return dataset


def update_data_table(df, table=None):
    """Updates the values of a DataTable from a DataFrame

    Parameters
    ----------
    df : TODO
    table : TODO

    Returns
    -------
    TODO

    """

    columns = [TableColumn(field=c, title=c) for c in df.columns]
    source = ColumnDataSource(df)
    if table is not None:
        table.columns = columns
        table.source = source
    else:
        return DataTable(source=source, columns=columns)
