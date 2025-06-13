from sqlalchemy import create_engine, Column, String, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker
import os

Base = declarative_base()


def get_db_engine():
    return create_engine('sqlite:///preprocessed_cache/data_cache.db', echo=False)


def init_db():
    engine = get_db_engine()
    if not os.path.exists('preprocessed_cache'):
        os.makedirs('preprocessed_cache')
    Base.metadata.create_all(engine)


class ProcessedData(Base):
    __tablename__ = 'processed_data'
    filename = Column(String, primary_key=True)
    country_code = Column(String)
    data = Column(LargeBinary)


def save_processed_data(filename, country_code, data_bytes):
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()

    existing = session.query(ProcessedData).filter_by(filename=filename).first()
    if existing:
        existing.country_code = country_code
        existing.data = data_bytes
    else:
        data = ProcessedData(
            filename=filename,
            country_code=country_code,
            data=data_bytes
        )
        session.add(data)
    session.commit()
    session.close()


def get_processed_data(filename):
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    result = session.query(ProcessedData).filter_by(filename=filename).first()
    session.close()
    return result.data if result else None


def list_processed_countries():
    engine = get_db_engine()
    Session = sessionmaker(bind=engine)
    session = Session()
    results = session.query(ProcessedData.country_code, ProcessedData.filename).distinct().all()
    session.close()
    return [{"country_code": r[0], "filename": r[1]} for r in results]


if __name__ == '__main__':
    init_db()
    print("Database initialized successfully!")