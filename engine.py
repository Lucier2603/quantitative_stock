
import os
import sys
import yaml

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

env = sys.argv[-1]
work_dir = os.getcwd()
CFG_FILEPATH = work_dir + '/config/config.yaml'
config = yaml.load(open(CFG_FILEPATH, 'r'), Loader=yaml.FullLoader)



stock_engine = create_engine(
    'mysql+mysqldb://{user}:{password}@{host}:{port}/{db}?charset={charset}'.format(
        db=config[env]['MySQL']['stock_engine_db'],
        host=config[env]['MySQL']['host'],
        port=config[env]['MySQL']['port'],
        user=config[env]['MySQL']['user'],
        password=config[env]['MySQL']['password'],
        charset="utf8"
    ), pool_pre_ping=True
)


class TAMP_SQL(object):
    """[sqlalchemy 封装]

    Args:
        object ([type]): [description]

    Returns:
        [type]: [description]
    """

    def __init__(self, db_engine):
        # 创建DBSession类型:
        self.DBSession = scoped_session(sessionmaker(bind=db_engine))
        self.session = self.DBSession()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            self.session.commit()
        except:
            self.session.rollback()
        finally:
            self.session.close()