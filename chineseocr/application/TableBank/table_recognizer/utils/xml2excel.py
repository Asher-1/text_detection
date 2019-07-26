# -*- coding: utf-8 -*-
import os
import bs4
import xlwt


class Xml2Excel(object):
    def __init__(self, logger=None, verbose=True):
        self._verbose = verbose
        self._logger = logger
        self._work_book = xlwt.Workbook(encoding='utf-8')
        self._sheet_index = 0

    @staticmethod
    def _read_file(path):
        with open(path, 'r+', encoding='UTF-8') as f:
            str = f.read()
        return str.strip().replace('\ufeff', '')

    def _parse_text_structure(self, structure, texts):
        soup = bs4.BeautifulSoup(structure, 'lxml')  # 解析html
        # elems=soup.findAll("table",{"class":"wikitable"})[0]#当需要进一步精确匹配时才使用
        table_ys_list = soup.findAll("tabular")
        for table_ys in table_ys_list:
            if self._verbose:
                if self._logger:
                    self._logger.info("#" * 50 + " start parsing sheet {} ".format(self._sheet_index) + "#" * 50)
            bgne_lb = []  # 设置一个列表用于接收数据
            table_trs = table_ys.findAll("tr")  # 获得表格中行的集合
            bghs = len(table_trs)

            for i in range(bghs):
                table_h = table_trs[i]
                z_lie = table_h.findAll(['tdy', 'thead', 'tdn'])  # 获得一行中列的集合
                bgls = len(z_lie)
                if self._verbose:
                    if self._logger:
                        self._logger.info("%s%s\t%s%s" % ("row ", i, " col number: ", bgls))
                bgnr_lb_h = []
                for ii in range(bgls):
                    if z_lie[ii].name == "tdy":
                        z_lie_value = z_lie[ii].getText() + "text"
                    elif z_lie[ii].name == "tdn":
                        z_lie_value = "empty"
                    elif z_lie[ii].name == "thead":
                        z_lie_value = "table head"
                    else:
                        raise NotImplementedError("cannot recognize unknown tags : {}".format(z_lie[ii].name))
                    bgnr_lb_h.append(z_lie_value.strip())  # 将单独一行中的数据写成一个列表
                bgne_lb.append(bgnr_lb_h)  # 将整行数据作为一个元素添加到bgne_lb列表中

            sheet = self._work_book.add_sheet("sheet" + str(self._sheet_index))
            self._sheet_index += 1
            for row, row_data in enumerate(bgne_lb):
                for col, value in enumerate(row_data):
                    sheet.write(row, col, value)

    def parse_xml_structure(self, data, texts=None):
        if isinstance(data, str):
            structure = self._read_file(data)
        elif isinstance(data, list):
            structure = "".join([d[0] for d in data])
        else:
            message = "the type of argument has not been implemented yet"
            if self._logger:
                self._logger.error(message)
            raise NotImplementedError(message)

        self._parse_text_structure(structure, texts)

    def save_excel(self, out_name=None):
        if out_name is None:
            self._logger.warning("out_name should be set before saving excel file")
            return
        if os.path.exists(out_name):
            os.remove(out_name)
        self._work_book.save(out_name)
        if self._logger:
            self._logger.info("save excel file to {}".format(out_name))
