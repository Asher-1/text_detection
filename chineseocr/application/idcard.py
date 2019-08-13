"""
身份证
"""
import re
from apphelper.image import union_rbox
from apphelper.tools import fuzzy_match


class idcard:
    """
    身份证结构化识别
    """

    def __init__(self, result):
        self.result = union_rbox(result, 0.2)
        self.N = len(self.result)
        self.res = {}
        self.full_name()
        self.sex()
        self.birthday()
        self.address()
        self.birthNo()
        self.organization()
        self.expiry_date()

    def full_name(self):
        """
        身份证姓名
        """
        name = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            # 匹配身份证姓名
            res = re.findall("姓名[\u4e00-\u9fa5]{1,4}", txt)
            if len(res) > 0:
                name['姓名'] = res[0].replace('姓名', '')
                self.res.update(name)
                break
        if "姓名" not in self.res.keys():
            txt = self.result[0]['text'].replace(' ', '')
            if len(txt) == 2:
                name['姓名'] = txt
            elif len(txt) == 3:
                name['姓名'] = txt[1:]
            elif len(txt) > 3:
                name['姓名'] = txt[2:]
            if 1 < len(name['姓名']) < 5:
                self.res.update(name)

    def sex(self):
        """
        性别女民族汉
        """
        sex = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            if '男' in txt:
                sex["性别"] = '男'
            elif '女' in txt:
                sex["性别"] = '女'

            ##性别女民族汉
            res = re.findall(".*民族[\u4e00-\u9fa5]+", txt)
            if len(res) > 0:
                sex["民族"] = res[0].split('民族')[-1]
                self.res.update(sex)
                break

    def birthday(self):
        """
        出生
        """
        birth = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ##出生年月
            res = re.findall('出生\d*年\d*月\d*日', txt)
            res = re.findall('\d*年\d*月\d*日', txt)

            if len(res) > 0:
                birth['出生'] = res[0].replace('出生', '').replace('年', '-').replace('月', '-').replace('日', '')
                self.res.update(birth)
                break

    def birthNo(self):
        """
        公民身份号码
        """
        No = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            txt = txt.replace(' ', '')
            ##公民身份号码
            res = re.findall('号码\d*[X|x]', txt)
            res += re.findall('号码\d*', txt)
            res += re.findall('\d{16,18}', txt)

            if len(res) > 0:
                No['公民身份号码'] = res[0].replace('号码', '')
                self.res.update(No)
                break

    def address(self):
        """
        身份证地址
        ##此处地址匹配还需完善
        """
        res_dict = {}
        start = 0
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(text=txt, target="公民身份号码")
            if t1 != "":
                if "住址" in res_dict.keys():
                    for j in range(start, i):
                        res_dict["住址"] += self.result[j]['text'].replace(' ', '')
                self.res.update(res_dict)
                break

            t1 = fuzzy_match(text=txt, target="住址")
            if t1 == "":
                continue

            res = re.findall(r"(?<=%s).+" % t1, txt)
            # key_word = ['住址', '省', '市', '县', '街', '村', '镇', '区', '城']
            if len(res) > 0:
                start = i + 1
                res_dict["住址"] = res[0].strip()

    # def address(self):
    #     """
    #     身份证地址
    #     ##此处地址匹配还需完善
    #     """
    #     add = {}
    #     addString = []
    #     for i in range(self.N):
    #         txt = self.result[i]['text'].replace(' ', '')
    #         txt = txt.replace(' ', '')
    #
    #         ##身份证地址
    #         if '住址' in txt or '省' in txt or '市' in txt or '县' in txt or '街' in txt or '村' in txt or "镇" in txt or "区" in txt or "城" in txt:
    #             addString.append(txt.replace('住址', ''))
    #
    #     if len(addString) > 0:
    #         add['身份证地址'] = ''.join(addString)
    #         self.res.update(add)

    def organization(self):
        """
        签发机关
        """
        No = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(text=txt, target="签发机关")
            if t1 == "":
                continue
            res = re.findall(r"(?<=%s).+" % t1, txt)

            if len(res) > 0:
                No['签发机关'] = res[0].strip()
                self.res.update(No)
                break

    def expiry_date(self):
        """
        有效期限
        """
        No = {}
        for i in range(self.N):
            txt = self.result[i]['text'].replace(' ', '')
            t1 = fuzzy_match(text=txt, target="有效期限")
            if t1 == "":
                continue
            res = re.findall(r"(?<=%s).+" % t1, txt)

            if len(res) > 0:
                No['有效期限'] = res[0].strip()
                self.res.update(No)
                break
