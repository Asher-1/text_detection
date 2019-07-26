from application.TableBank.table_detector.table_detector import TableDetector
from application.TableBank.table_detector.base import cfgs

if __name__ == '__main__':
    TR = TableDetector()
    TR.detect_tables(im_or_folder=cfgs.TEST_PATH)
