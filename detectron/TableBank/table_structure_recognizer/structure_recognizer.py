#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from itertools import repeat
from .utils.xml2excel import Xml2Excel

import os
import sys
workspace = os.path.basename(os.path.dirname(__file__))
sys.path.append(workspace)

from .onmt import opts as opts
from .onmt.utils.misc import split_corpus
from .onmt.utils.parse import ArgumentParser
from .onmt.utils.logging import init_logger
from .onmt.translate.translator import build_translator

from .base.configs import cfgs


class StructureRecognizer(object):
    def __init__(self):
        self.cfgs = cfgs
        parser = self._get_parser()
        self._opt = parser.parse_args()
        self._logger = init_logger(self._opt.log_file)
        self._transformer = Xml2Excel(logger=self._logger)

    @staticmethod
    def _get_parser():
        parser = ArgumentParser(description='structure_recognizer.py')
        opts.config_opts(parser)
        opts.translate_opts(parser, cfgs)
        return parser

    def recognize_structure(self, texts=None):
        ArgumentParser.validate_translate_opts(self._opt)

        translator = build_translator(self._opt, report_score=True, logger=self._logger)
        src_shards = split_corpus(self._opt.src, self._opt.shard_size)
        tgt_shards = split_corpus(self._opt.tgt, self._opt.shard_size) \
            if self._opt.tgt is not None else repeat(None)
        shard_pairs = zip(src_shards, tgt_shards)

        for i, (src_shard, tgt_shard) in enumerate(shard_pairs):
            self._logger.info("Translating shard %d." % i)
            all_scores, all_predictions = translator.translate(
                src=src_shard,
                tgt=tgt_shard,
                src_dir=self._opt.src_dir,
                batch_size=self._opt.batch_size,
                attn_debug=self._opt.attn_debug,
            )
            print("all_scores : ", all_scores)
            print("all_predictions : ", all_predictions)
            self._transformer.parse_xml_structure(all_predictions, texts=texts)
            # self._transformer.parse_xml_structure(cfgs.PRE_PATH, texts=texts)

    def save_excel(self, out_name=cfgs.TABLE_PATH):
        self._transformer.save_excel(out_name=out_name)
