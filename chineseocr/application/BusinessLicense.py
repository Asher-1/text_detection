"""
身份证
"""
from apphelper.image import union_rbox
from apphelper.tools import fuzzy_match
import re
import json


class BusinessLicense:
    """
    营业执照结构化识别
    """

    def __init__(self, result):
        self.result = union_rbox(result, 0.2)
        json.dump(self.result, open("result.json", "w"), ensure_ascii=False, indent=4)
        self.N = len(self.result)
        self.res = {}
        self.license_number()
        self.registered_number()
        self.name()
        self.registered_capital()
        self.type()
        self.date_setup()
        self.legal_representatives()
        self.term_operation()
        self.scope_management()
        self.address()

    def license_number(self):
        """
        证照编号
        """
        No = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(txt, target="注册号")
            t2 = fuzzy_match(txt, target="证照编号")
            if t1 in txt and t1 != "":
                res = re.findall(r"(?<=%s).+" % t1, txt)
                key = "注册号"
            elif t2 in txt and t2 != "":
                res = re.findall(r"(?<=%s).+" % t2, txt)
                key = "证照编号"
            else:
                continue
            if len(res) > 0:
                text_number = res[0].strip()
                no = list(filter(str.isdigit, text_number))
                no = "".join(no).strip()
                No[key] = no
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
            t1 = fuzzy_match(txt, target="统一社会信用代码")
            if t1 in txt and t1 != "":
                res = re.findall(r"(?<=%s).+" % t1, txt)
                if len(res) > 0:
                    no = list(filter(str.isalnum, res[0].strip()))
                    no = "".join(no).strip()
                    if len(no) > 16:
                        No["统一社会信用代码"] = no
                        self.res.update(No)
                        break
                no = list(filter(str.isalnum, self.result[i + 2]['text'].replace(' ', '')))
                no = "".join(no).strip()
                if len(no) == 17:
                    no += 'X'
                No["统一社会信用代码"] = no
                self.res.update(No)
                break

    def name(self):
        """
        企业名称
        """
        res_dict = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(txt, target="名称")
            t2 = fuzzy_match(text=txt, target="注册资本", max_error_num=1)
            if t1 == "":
                continue
            else:
                pattern = r"%s(.*)%s" % (t1, t2)
            res = re.findall(pattern, txt)

            if len(res) > 0:
                res_dict['名称'] = res[0].strip()
                self.res.update(res_dict)
                break

    def registered_capital(self):
        """
        注册资本
        """
        res_dict = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(text=txt, target="注册资本", max_error_num=1)
            if t1 == "":
                continue
            res = re.findall(r"(?<=%s).+" % t1, txt)
            if len(res) > 0:
                res_dict["注册资本"] = res[0].strip()
                self.res.update(res_dict)
                break

    def type(self):
        """
        企业类型
        """
        res_dict = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')

            t1 = fuzzy_match(txt, target="类型")
            t2 = fuzzy_match(txt, target="成立日期")
            if t1 == "":
                continue
            else:
                pattern = r"%s(.*)%s" % (t1, t2)
            res = re.findall(pattern, txt)

            if len(res) > 0:
                res_dict['企业类型'] = res[0].strip()
                self.res.update(res_dict)
                break

    def date_setup(self):
        """
        成立日期
        """
        res_dict = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(txt, target="成立日期")
            if t1 == "":
                continue
            res = re.findall(r"(?<=%s).+" % t1, txt)
            if len(res) > 0:
                res_dict["成立日期"] = res[0].strip()
                self.res.update(res_dict)
                break

    def legal_representatives(self):
        """
        法定代表人
        """
        res_dict = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')

            t1 = fuzzy_match(txt, target="法定代表人")
            t2 = fuzzy_match(text=txt, target="营业期限", max_error_num=1)
            if t1 == "":
                continue
            else:
                pattern = r"%s(.*)%s" % (t1, t2)
            res = re.findall(pattern, txt)

            # 法定代表人
            if len(res) > 0:
                res_dict['法定代表人'] = res[0].strip()
                self.res.update(res_dict)
                break

    def term_operation(self):
        """
        营业期限
        """
        res_dict = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(text=txt, target="营业期限", max_error_num=1)
            if t1 == "":
                continue
            # 营业期限
            res = re.findall(r"(?<=%s).+" % t1, txt)
            if len(res) > 0:
                res_dict["营业期限"] = res[0].strip()
                self.res.update(res_dict)
                break

    def scope_management(self):
        res_dict = {}
        start = 0
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(text=txt, target="登记机关")
            if t1 != "":
                if "经营范围" in res_dict.keys():
                    for j in range(start, i):
                        res_dict["经营范围"] += self.result[j]['text'].replace(' ', '')
                self.res.update(res_dict)
                break

            t1 = fuzzy_match(text=txt, target="经营范围", max_error_num=1)
            t2 = fuzzy_match(txt, target="住所")
            if t1 == "":
                continue
            else:
                pattern = r"%s(.*)%s" % (t1, t2)
            res = re.findall(pattern, txt)
            if len(res) > 0:
                start = i + 1
                res_dict["经营范围"] = res[0].strip()

    def address(self):
        """
        住所
        """
        res_dict = {}
        address = []
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(txt, target="住所")
            if t1 == "":
                continue

            res = re.findall(r"(?<=%s).+" % t1, txt)
            if len(res) > 0:
                key_word = ['住所', '省', '市', '县', '街', '村', '镇', '区', '城']
                if any(key in key_word for key in res[0]):
                    address.append(res[0].replace('住所', ''))
                if len(address) > 0:
                    res_dict['住所'] = ''.join(address)
                    self.res.update(res_dict)
                    break
