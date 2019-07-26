from TableBank.table_structure_recognizer.structure_recognizer import StructureRecognizer
from TableBank.table_structure_recognizer.structure_recognizer import cfgs

if __name__ == '__main__':
    SR = StructureRecognizer()
    SR.recognize_structure()
    SR.save_excel(cfgs.TABLE_PATH)
