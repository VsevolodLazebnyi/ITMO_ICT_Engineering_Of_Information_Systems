from sqlalchemy import create_engine, Column, String, LargeBinary
from sqlalchemy.orm import declarative_base, sessionmaker
import os

Base = declarative_base()
engine = create_engine('sqlite:///preprocessed_cache/data_cache.db', echo=False)
Session = sessionmaker(bind=engine)

class ProcessedData(Base):
    __tablename__ = 'processed_data'
    filename = Column(String, primary_key=True)
    country_code = Column(String)
    data = Column(LargeBinary)

def init_db():
    if not os.path.exists('preprocessed_cache'):
        os.makedirs('preprocessed_cache')
    Base.metadata.create_all(engine)

def save_processed_data(filename, country_code, data_bytes):
    session = Session()
    data = ProcessedData(
        filename=filename,
        country_code=country_code,
        data=data_bytes
    )
    session.merge(data)
    session.commit()

def get_processed_data(filename):
    session = Session()
    result = session.query(ProcessedData).filter_by(filename=filename).first()
    return result.data if result else None

if __name__ == '__main__':
    init_db()