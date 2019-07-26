from application.TableBank.table_recognizer.structure_recognizer import TableRecognizer
from application.TableBank.table_recognizer.base import cfgs

if __name__ == '__main__':
    TR = TableRecognizer()
    TR.recognize_structure()
    TR.save_excel(cfgs.TABLE_PATH)
