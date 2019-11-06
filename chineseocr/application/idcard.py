"""
身份证
"""
import re
import json
from apphelper.image import union_rbox
from apphelper.tools import fuzzy_match


class idcard:
    """
    身份证结构化识别
    """

    def __init__(self, result):
        self.result = union_rbox(result, 0.2)
        # json.dump(self.result, open("result.json", "w"), ensure_ascii=False, indent=4)
        self.N = len(self.result)
        self.res = {}
        self.organization()
        self.expiry_date()
        self.full_name()
        self.sex()
        self.birthday()
        self.address()
        self.birthNo()

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
            for j in range(self.N):
                txt = self.result[j]['text'].replace(' ', '')
                if len(txt) > 1:
                    fuz = fuzzy_match(text=txt, target='姓名')
                    if '签发机关' in self.res.keys() or '有效期限' in self.res.keys():
                        if fuz == '':
                            break
                    else:
                        txt = txt.replace(fuz, '')
                        if len(txt) <= 4:
                            name['姓名'] = txt
                        else:
                            name['姓名'] = txt[-4:]
                        self.res.update(name)
                        break

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
            if '年' in txt or '月' in txt or '日' in txt:
                # res = re.findall('\d*年\d*月\d*日', txt)
                txt = txt.replace(' ', '')
                t1 = fuzzy_match(text=txt, target="出生")
                res = re.findall(r'\d{4}(.?)\d{1,2}(.?)\d{1,2}(.?)', txt)
                if len(res) > 0 and len(res[0]) > 0:
                    if len(t1) > 0:
                        txt = txt[txt.index(t1[-1]) + 1:]
                    txt = txt.replace(res[0][-1], '')
                    for i in range(len(res[0]) - 1):
                        txt = txt.replace(res[0][i], '-')
                    birth['出生'] = txt
                    self.res.update(birth)
                    break

            # if len(res) > 0:
            #     birth['出生'] = res[0].replace('出生', '').replace('年', '-').replace('月', '-').replace('日', '')
            #     self.res.update(birth)
            #     break

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
