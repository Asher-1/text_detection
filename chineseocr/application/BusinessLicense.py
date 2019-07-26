"""
身份证
"""
from apphelper.image import union_rbox
import re


class BusinessLicense:
    """
    身份证结构化识别
    """

    def __init__(self, result):
        self.result = union_rbox(result, 0.2)
        self.N = len(self.result)
        self.res = {}
        self.license_number()
        self.registered_number()
        self.name_capital()
        self.type_date_setup()
        self.legal_term_operation()
        self.scope_address()

    def license_number(self):
        """
        证照编号
        """
        No = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            res = re.findall(r"(?<=证照编号).+", txt)
            if len(res) > 0:
                text_number = res[0].strip()
                no = list(filter(str.isdigit, text_number))
                no = "".join(no).strip()
                No["统一社会信用代码"] = no
                self.res.update(No)
                break

    def registered_number(self):
        """
        统一社会信用代码
        """
        No = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            if '统一社会信用代码' in txt:
                no = list(filter(str.isalnum, self.result[i + 2]['text'].replace(' ', '')))
                no = "".join(no).strip()
                if len(no) == 17:
                    no += 'X'
                No["统一社会信用代码"] = no
                self.res.update(No)
                break

    def name_capital(self):
        """
        企业名称 和 注册资本
        """
        res_dict = {}
        for i in range(self.N):
            if "名称" in res_dict.keys() and "注册资本" in res_dict.keys():
                self.res.update(res_dict)
                break

            txt = self.result[i]['text'].replace(' ', '')

            # 企业名称
            res = re.findall(r".*名称(.*?)注册资本", txt)
            if len(res) > 0:
                res_dict['名称'] = res[0].strip()

            # 注册资本
            res = re.findall(r"(?<=注册资本).+", txt)
            if len(res) > 0:
                res_dict["注册资本"] = res[0].strip()

    def type_date_setup(self):
        """
        企业类型 和 成立日期
        """
        res_dict = {}
        for i in range(self.N):
            if "类型" in res_dict.keys() and "成立日期" in res_dict.keys():
                self.res.update(res_dict)
                break
            txt = self.result[i]['text'].replace(' ', '')

            # 企业类型
            res = re.findall(r".*类型(.*?)成立日期", txt)
            if len(res) > 0:
                res_dict['类型'] = res[0].strip()

            # 成立日期
            res = re.findall(r"(?<=成立日期).+", txt)
            if len(res) > 0:
                res_dict["成立日期"] = res[0].strip()

    def legal_term_operation(self):
        """
        法定代表人 和 营业期限
        """
        res_dict = {}
        for i in range(self.N):
            if "法定代表人" in res_dict.keys() and "营业期限" in res_dict.keys():
                self.res.update(res_dict)
                break
            txt = self.result[i]['text'].replace(' ', '')

            # 企业类型
            res = re.findall(r".*法定代表人(.*?)营业期限", txt)
            if len(res) > 0:
                res_dict['法定代表人'] = res[0].strip()

            # 成立日期
            res = re.findall(r"(?<=营业期限).+", txt)
            if len(res) > 0:
                res_dict["营业期限"] = res[0].strip()

    def scope_address(self):
        """
        经营范围 和 住所
        """
        res_dict = {}
        address = []
        start = 0
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            res = re.findall(r".*经营范围(.*?)住所", txt)
            if len(res) > 0:
                start = i + 1
                res_dict["经营范围"] = res[0].strip()

            # 成立日期
            res = re.findall(r"(?<=住所).+", txt)
            if len(res) > 0:
                key_word = ['住所', '省', '市', '县', '街', '村', '镇', '区', '城']
                if any(key in key_word for key in res[0]):
                    address.append(res[0].replace('住所', ''))
                if len(address) > 0:
                    res_dict['住所'] = ''.join(address)
            if "登记机关" in txt:
                if "经营范围" in res_dict.keys():
                    for j in range(start, i):
                        res_dict["经营范围"] += self.result[j]['text'].replace(' ', '')
                    self.res.update(res_dict)
                break
