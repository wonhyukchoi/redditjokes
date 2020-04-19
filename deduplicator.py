import pandas as pd
import numpy as np
from SetSimilaritySearch import all_pairs
from collections import defaultdict
from itertools import chain
from konlpy.tag import Okt


class Deduplicator:

    def deduplicate_dataframe(self, df: pd.DataFrame, target_column: str,
                              mode='okt', gram_len=3, criterion='jaccard', threshold=0.4):
        deduplicate_indices = self._get_deduplicated_indices(text_list=df[target_column],
                                                             mode=mode, gram_len=gram_len,
                                                             criterion=criterion, threshold=threshold)
        return df.iloc[deduplicate_indices]

    def deduplicate_texts(self, text_list: list, mode='okt', gram_len=3, criterion='jaccard', threshold=0.4):
        deduplicate_indices = self._get_deduplicated_indices(text_list=text_list,
                                                             mode=mode, gram_len=gram_len,
                                                             criterion=criterion, threshold=threshold)
        return [text for n, text in enumerate(text_list) if n in deduplicate_indices]

    def _get_deduplicated_indices(self, text_list: list, mode='okt', gram_len=3, criterion='jaccard', threshold=0.4):
        duplicate_sets = self._get_duplicate_sets(text_list=text_list, mode=mode, gram_len=gram_len,
                                                  criterion=criterion, threshold=threshold)

        duplicate_articles = list(chain(* duplicate_sets))
        duplicate_representatives = [grouped_duplicates[0] for grouped_duplicates in duplicate_sets]

        non_duplicates = [i for i in range(len(text_list)) if i not in duplicate_articles]

        output_indices = non_duplicates + duplicate_representatives
        return output_indices

    def _get_duplicate_sets(self, text_list: list, mode='okt', gram_len=3, criterion='jaccard', threshold=0.4):
        text_similarities = self._find_text_similarities(text_list=text_list, mode=mode,
                                                         gram_len=gram_len, criterion=criterion,
                                                         threshold=threshold)
        return list(self._group_duplicates(text_similarities).values())

    def _find_text_similarities(self, text_list: list, mode='okt', gram_len=3,
                                criterion='jaccard', threshold=0.4) -> list:
        """
        여러개의 원문들의 유사도를 제공합니다.
        :param text_list: [[원문], [원문], ...] (list of strings)
        :param criterion: jaccard 또는 cosine
        :param threshold: hyperparameter, 낮으면 낮을수록 유사도를 높임
        :return: n x 3 list.
        column 1, 2 : 실제 원문들의 인덱스값
        column 3: 유사도 값
        """
        split_texts = self._split_texts(text_list=text_list, mode=mode)
        ngram_splits = [self._get_ngrams(one_text, gram_len=gram_len)
                        for one_text in split_texts]
        pairs = all_pairs(ngram_splits, similarity_func_name=criterion,
                          similarity_threshold=threshold)
        return list(pairs)

    @staticmethod
    def _split_texts(text_list: list, mode='simple') -> list:
        """
        여러개의 strings들의 list를 split해서 돌려줌
        :param text_list: ['국회는 오늘 ...', 'LG전자 어쩌구' ...]
        :param mode: simple (쉼표구분), okt(명사만 추출)
        :return: (['국회', '오늘', ...], ['LG전자', '어쩌구', ...])
        """
        if mode == 'simple':
            return [text_row.split() for text_row in text_list]
        elif mode == 'okt':
            okt = Okt()
            return [okt.nouns(text_row) for text_row in text_list]
        else:
            raise KeyError(f"Expected simple or okt, got {mode}")

    @staticmethod
    def _get_ngrams(word_list: tuple, gram_len=3) -> set:
        """
        :param word_list: n-gram을 발췌할 단어들의 목록
        ['디시인사이드', '갤러리', '갤럼', '자전거'...]
        :param gram_len: n-gram의 "n" 값
        :return: n-gram 단위의 string으로 묶인 unordered set
        {'디시인사이드 갤러리 갤럼', '갤러리 갤럼 자전거', ...}
        """
        ngrams_list = tuple(' '.join(word_list[i: i + gram_len]) for
                            i in range(len(word_list) - gram_len))
        return set(ngrams_list)

    @staticmethod
    def _group_duplicates(sim_results):
        """
        dfs를 통해서 유사도 높은 원문끼리 하나의 set으로 묶음
        :param sim_results: 
        :return: 
        """

        def dfs(adj_list, visited, vertex, result, key):
            visited.add(vertex)
            result[key].append(vertex)
            for neighbor in adj_list[vertex]:
                if neighbor not in visited:
                    dfs(adj_list, visited, neighbor, result, key)

        # 유사도를 nx2 array로 변경
        edges = np.array(sim_results, dtype=int)[:, :2]

        # 유사도를 adjacency list로 변경
        adj_list = defaultdict(list)
        for x, y in edges:
            adj_list[x].append(y)
            adj_list[y].append(x)

        # 연결된 부분들끼리 연결
        result = defaultdict(list)
        visited = set()
        for vertex in adj_list:
            if vertex not in visited:
                dfs(adj_list, visited, vertex, result, vertex)

        return result
