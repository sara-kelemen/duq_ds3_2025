import sqlalchemy


def db(path: str = 'data/patent_npi_db.sqlite'):
    engine = sqlalchemy.create_engine('sqlite:///' + path)
    conn = engine.connect()
    return conn