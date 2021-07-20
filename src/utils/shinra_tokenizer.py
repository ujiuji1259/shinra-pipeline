import re
import unicodedata

from transformers import AutoTokenizer

def replace_unk(tokens, subwords):
    unk_subwords = []
    for token, subword in zip(tokens, subwords):
        if len(subword) == 1 and subword[0] == "[UNK]":
            unk_subwords += [token]
            continue

        unk_subwords += subword
    return unk_subwords


# "##:"など、"##"がsubwordのprefixじゃないとき用
def replace_subword_prefix_for_ambiguous(token_list, word_tokens):
    original_sent = "".join(word_tokens)
    results = []
    amb_indexes = []
    for i in range(len(token_list)):
        if token_list[i].startswith("##"):
            results.append(token_list[i][2:])
            amb_indexes.append(i)
            continue
        results.append(token_list[i])
    assert len(amb_indexes) > 0

    # たいていambiguousな「##」は一つなので、まず一個ずつ確認
    for idx in amb_indexes:
        tmp = results[:idx] + ["##" + results[idx]] + results[idx + 1 :]
        if "".join(tmp) == original_sent:
            return tmp

    # それでも無理なら全探索
    # 再帰だと「##」が多すぎると深くなりすぎてエラー吐く
    # のでforで実装
    for j in range(1 << len(amb_indexes)):
        tmp = []
        prev_index = 0
        for idx in range(len(amb_indexes)):
            amb_index = amb_indexes[idx]
            tmp.extend(results[prev_index:amb_index])
            mask = 1 << idx
            if j & mask:
                tmp.append("##" + results[amb_index])
            else:
                tmp.append(results[amb_index])
            prev_index = amb_index + 1
        tmp.extend(results[amb_index + 1 :])

        if "".join(tmp) == original_sent:
            return tmp

    assert False, "ambiguous ### token"


# subwordのprefix（##）を消す
def replace_subword_prefix(token_list, word_tokens):
    results = []
    is_ambiguous = False
    amb_indexes = []
    for i in range(len(token_list)):
        # "###"は"###"の場合と"#"にsubword prefixが付いている場合の２パターン
        is_three = token_list[i] == "###"
        is_prefix = i > 0 and token_list[i - 1] == "###"
        is_prefix_amb = (
            i > 0
            and i < len(token_list) - 1
            and token_list[i - 1] != "###"
            and token_list[i + 1] != "###"
        )
        is_prefix_amb |= i == len(token_list) - 1 and token_list[i - 1] != "###"
        if (not is_three and token_list[i].startswith("##")) or (
            is_three and is_prefix
        ):
            results.append(token_list[i][2:])
            continue
        elif is_three and is_prefix_amb:
            is_ambiguous = True
            amb_indexes.append(i)
            results.append(token_list[i][2:])
            continue

        results.append(token_list[i])

    original_sent = "".join(word_tokens)
    if is_ambiguous:
        # 曖昧な"###"を、全パターン試す
        for j in range(1 << len(amb_indexes)):
            tmp = []
            prev_index = 0
            for idx in range(len(amb_indexes)):
                amb_index = amb_indexes[idx]
                tmp.extend(results[prev_index:amb_index])
                mask = 1 << idx
                if j & mask:
                    tmp.append("##" + results[amb_index])
                else:
                    tmp.append(results[amb_index])
                prev_index = amb_index + 1
            tmp.extend(results[amb_index + 1 :])

            if "".join(tmp) == original_sent:
                # 元の文と一致したら終了
                return tmp
    else:
        if "".join(results) == original_sent:
            return results

    # "##:"など、"##""から始まってもsubwordじゃ無い場合があるので、
    # そういう場合は"##"をつけるパターンとつけないパターンを全探索する
    return replace_subword_prefix_for_ambiguous(token_list, word_tokens)


def tokenize_sent(line, tokenizer):
    tokenizer.do_word_tokenize = True
    tokenizer.do_subword_tokenize = False
    tokens = tokenizer.tokenize(line)
    assert len(tokens) > 0

    tokenizer.do_word_tokenize = False
    tokenizer.do_subword_tokenize = True
    tokens = [re.sub("\s", "", t) for t in tokens]
    subwords = [tokenizer.tokenize(t) for t in tokens]
    flatten_subwords = [s for sub in subwords for s in sub]
    unk_subwords = replace_unk(tokens, subwords)

    normalized_tokens = replace_subword_prefix(unk_subwords, tokens)
    offsets = []
    token_offset = 0
    text_offset = 0
    intoken_offset = 0

    token_len = 1
    normalized_text = unicodedata.normalize("NFKC", line[text_offset])
    while text_offset + token_len <= len(line) and token_offset < len(
        normalized_tokens
    ):
        # 空白用。たまにNFKCで先頭に空白が入ったりする
        # 先頭以外にも入ったりするのでreを使う
        # normalized_text = re.sub(r'^\s', '', normalized_text)
        normalized_text = re.sub("\s", "", normalized_text)
        # 空白だけのとき用
        if normalized_text == "":
            token_len = 1
            text_offset += 1
            intoken_offset = 0
            normalized_text = unicodedata.normalize(
                "NFKC", line[text_offset + token_len - 1]
            )
            continue

        # tokenの方が長い場合. i.e. 分解された合成文字の途中でtokenが途切れている場合
        if (
            normalized_text.startswith(normalized_tokens[token_offset])
            and normalized_text != normalized_tokens[token_offset]
        ):
            offsets.append((text_offset, text_offset + token_len))
            normalized_text = normalized_text[len(normalized_tokens[token_offset]) :]
            token_offset += 1
            text_offset += token_len - 1
            token_len = 1
            original_normalized_text = unicodedata.normalize(
                "NFKC", line[text_offset : text_offset + token_len]
            )
            intoken_offset = len(original_normalized_text) - len(normalized_text)
            continue

        # マッチ
        if normalized_text == normalized_tokens[token_offset]:
            offsets.append((text_offset, text_offset + token_len))
            text_offset += token_len
            token_offset += 1
            token_len = 1
            intoken_offset = 0
            # 最後の文字用
            if text_offset + token_len - 1 < len(line):
                normalized_text = unicodedata.normalize(
                    "NFKC", line[text_offset + token_len - 1]
                )
                # normalized_text = re.sub(r'^\s', '', normalized_text)
        else:
            # tokenの方が長い場合
            token_len += 1
            # 非基幹文字のみの場合、文字順が変わる場合があるので、normalized_textを増やす場合ももう一度全てnormalizeする
            normalized_text = unicodedata.normalize(
                "NFKC", line[text_offset : text_offset + token_len]
            )[intoken_offset:]

    if text_offset < len(line):
        print(flatten_subwords, offsets)
        assert (
            unicodedata.normalize("NFKC", line[text_offset:]).strip() == ""
        ), "空白文字以外の文字が残っています"
        text_offset = len(line)

    assert text_offset == len(line) and token_offset == len(
        flatten_subwords
    ), "テキストかトークンが残っています"

    return flatten_subwords, offsets

if __name__ == "__main__":
    texts = "こんにちは．わたしは宇宙人なのです"
    tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    print(tokenize_sent(texts, tokenizer))