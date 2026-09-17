import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset


ANSWER_LETTERS = ["A", "B", "C", "D"]
COMMONSENSEQA_ANSWER_LETTERS = ["A", "B", "C", "D", "E"]


def _normalize_answer(raw_answer: Any, answer_letters: Sequence[str] = ANSWER_LETTERS) -> int:
    if isinstance(raw_answer, int):
        return raw_answer

    if isinstance(raw_answer, str):
        candidate = raw_answer.strip().upper()
        if candidate in answer_letters:
            return list(answer_letters).index(candidate)
        if candidate.isdigit():
            return int(candidate)

    raise ValueError(f"Unsupported answer format: {raw_answer!r}")


def _format_user_prompt(example: Dict[str, Any]) -> Tuple[str, str]:
    question = example["question"].strip()
    choices = example["choices"]
    subject = str(example.get("subject", "unknown_subject")).replace("_", " ").strip()
    answer_idx = _normalize_answer(example["answer"])
    answer_letter = ANSWER_LETTERS[answer_idx]
    answer_text = choices[answer_idx].strip()

    choice_lines = [f"{letter}. {text.strip()}" for letter, text in zip(ANSWER_LETTERS, choices)]
    user_prompt = (
        f"Subject: {subject}\n"
        f"Question: {question}\n"
        "Choices:\n"
        f"{chr(10).join(choice_lines)}\n\n"
        "Return the single correct answer in the format `LETTER. answer text`."
    )
    assistant_response = f"{answer_letter}. {answer_text}"
    return user_prompt, assistant_response


def _format_gsm8k_prompt(example: Dict[str, Any], system_prompt: str) -> Tuple[str, str]:
    question = example["question"].strip()
    answer = example["answer"].strip()
    user_prompt = (
        "Solve the following grade-school math word problem.\n\n"
        f"Question: {question}"
    )
    return user_prompt, answer


def _format_math_prompt(example: Dict[str, Any], system_prompt: str) -> Tuple[str, str]:
    problem = str(example["problem"]).strip()
    solution = str(example["solution"]).strip()
    subject = str(example.get("type", "")).strip()
    level = str(example.get("level", "")).strip()

    metadata = []
    if subject:
        metadata.append(f"Subject: {subject}")
    if level:
        metadata.append(f"Difficulty: {level}")
    metadata_text = "\n".join(metadata)
    if metadata_text:
        metadata_text += "\n"

    user_prompt = (
        f"{metadata_text}"
        "Solve the following competition math problem. Show your reasoning and put the final answer in "
        "\\boxed{}.\n\n"
        f"Problem: {problem}"
    )
    return user_prompt, solution


def _normalize_choice_pairs(choices: Any) -> List[Tuple[str, str]]:
    if isinstance(choices, dict):
        labels = choices.get("label") or choices.get("labels")
        texts = choices.get("text") or choices.get("texts")
        if labels is None or texts is None:
            raise ValueError(f"Unsupported choices dict format: {choices!r}")
        return [(str(label).strip().upper(), str(text).strip()) for label, text in zip(labels, texts)]

    if isinstance(choices, list):
        if all(isinstance(item, dict) for item in choices):
            pairs = []
            for idx, item in enumerate(choices):
                label = item.get("label") or item.get("key") or COMMONSENSEQA_ANSWER_LETTERS[idx]
                text = item.get("text") or item.get("value") or item.get("answer")
                if text is None:
                    raise ValueError(f"Unsupported choice item format: {item!r}")
                pairs.append((str(label).strip().upper(), str(text).strip()))
            return pairs
        return [
            (COMMONSENSEQA_ANSWER_LETTERS[idx], str(text).strip())
            for idx, text in enumerate(choices)
        ]

    raise ValueError(f"Unsupported choices format: {choices!r}")


def _format_commonsenseqa_prompt(example: Dict[str, Any], system_prompt: str) -> Tuple[str, str]:
    question = str(example["question"]).strip()
    choice_pairs = _normalize_choice_pairs(example["choices"])
    labels = [label for label, _ in choice_pairs]
    answer = example.get("answerKey", example.get("answer"))
    if answer is None:
        raise ValueError("CommonsenseQA example does not contain answerKey/answer.")
    answer_label = str(answer).strip().upper()
    if answer_label.isdigit():
        answer_idx = _normalize_answer(answer_label, labels)
        answer_label = labels[answer_idx]
    if answer_label not in labels:
        raise ValueError(f"CommonsenseQA answer {answer!r} not found in choices {labels!r}.")
    answer_text = dict(choice_pairs)[answer_label]

    choice_lines = [f"{label}. {text}" for label, text in choice_pairs]
    user_prompt = (
        f"Question: {question}\n"
        "Choices:\n"
        f"{chr(10).join(choice_lines)}\n\n"
        "Return the single correct answer in the format `LETTER. answer text`."
    )
    assistant_response = f"{answer_label}. {answer_text}"
    return user_prompt, assistant_response


def _normalize_messages(raw_messages: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in raw_messages:
        if not isinstance(message, dict):
            raise ValueError(f"Unsupported message format: {message!r}")

        role = str(message.get("role", "")).strip().lower()
        content = str(message.get("content", "")).strip()
        if not role or not content:
            continue
        normalized.append({"role": role, "content": content})

    if not normalized:
        raise ValueError("No valid messages found in example.")

    return normalized


def _render_messages(tokenizer, messages, add_generation_prompt: bool) -> str:
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    role_labels = {
        "system": "System",
        "user": "User",
        "assistant": "Assistant",
    }
    segments = []
    for message in messages:
        role = role_labels.get(message["role"], message["role"].title())
        segments.append(f"{role}: {message['content'].strip()}")

    if add_generation_prompt:
        segments.append("Assistant:")

    return "\n\n".join(segments)


def _format_example(example: Dict[str, Any], dataset_name: str, system_prompt: str) -> Tuple[str, str]:
    normalized = dataset_name.lower()
    if "gsm8k" in normalized:
        return _format_gsm8k_prompt(example, system_prompt)
    if "math-lighteval" in normalized or "competition_math" in normalized:
        return _format_math_prompt(example, system_prompt)
    if "commonsense_qa" in normalized or "commonsenseqa" in normalized:
        return _format_commonsenseqa_prompt(example, system_prompt)
    if "mmlu" in normalized:
        return _format_user_prompt(example)
    raise ValueError(f"Unsupported dataset format for {dataset_name!r}")


def _infer_dataset_format(
    example: Dict[str, Any],
    dataset_name: str,
    dataset_format: str,
    messages_field: str,
) -> str:
    normalized_format = str(dataset_format or "auto").lower()
    if normalized_format != "auto":
        return normalized_format

    if messages_field in example:
        return "messages"

    normalized_name = dataset_name.lower()
    if "gsm8k" in normalized_name:
        return "gsm8k"
    if "math-lighteval" in normalized_name or "competition_math" in normalized_name:
        return "math"
    if "commonsense_qa" in normalized_name or "commonsenseqa" in normalized_name:
        return "commonsenseqa"
    if "mmlu" in normalized_name:
        return "mmlu"

    raise ValueError(
        "Could not infer dataset format. Set dataset.format explicitly to one of "
        "'gsm8k', 'math', 'mmlu', 'commonsenseqa', or 'messages'."
    )


def _build_messages(
    example: Dict[str, Any],
    dataset_name: str,
    system_prompt: str,
    dataset_format: str,
    messages_field: str,
    prepend_system_prompt: bool,
) -> List[Dict[str, str]]:
    resolved_format = _infer_dataset_format(example, dataset_name, dataset_format, messages_field)
    if resolved_format == "messages":
        if messages_field not in example:
            raise KeyError(
                f"Configured messages field {messages_field!r} not found in dataset example keys "
                f"{list(example.keys())!r}."
            )
        messages = _normalize_messages(example[messages_field])
        has_system_message = bool(messages) and messages[0]["role"] == "system"
        if prepend_system_prompt and system_prompt.strip() and not has_system_message:
            return [{"role": "system", "content": system_prompt.strip()}] + messages
        return messages

    if resolved_format == "gsm8k":
        user_prompt, assistant_response = _format_gsm8k_prompt(example, system_prompt)
    elif resolved_format == "math":
        user_prompt, assistant_response = _format_math_prompt(example, system_prompt)
    elif resolved_format == "commonsenseqa":
        user_prompt, assistant_response = _format_commonsenseqa_prompt(example, system_prompt)
    elif resolved_format == "mmlu":
        user_prompt, assistant_response = _format_user_prompt(example)
    else:
        user_prompt, assistant_response = _format_example(example, dataset_name, system_prompt)
    messages: List[Dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt.strip()})
    messages.extend(
        [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response},
        ]
    )
    return messages


def _assistant_target_spans(tokenizer, messages: Sequence[Dict[str, str]]) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        if idx == 0:
            prompt_len = 0
        else:
            prompt_text = _render_messages(tokenizer, messages[:idx], add_generation_prompt=True)
            prompt_len = len(tokenizer(prompt_text, add_special_tokens=False)["input_ids"])

        full_text = _render_messages(tokenizer, messages[: idx + 1], add_generation_prompt=False)
        full_len = len(tokenizer(full_text, add_special_tokens=False)["input_ids"])
        if full_len > prompt_len:
            spans.append((prompt_len, full_len))

    return spans


def _assistant_target_char_spans(
    tokenizer,
    messages: Sequence[Dict[str, str]],
) -> List[Tuple[int, int]]:
    spans: List[Tuple[int, int]] = []
    for idx, message in enumerate(messages):
        if message["role"] != "assistant":
            continue

        content = message["content"].strip()
        if not content:
            continue

        prefix_text = _render_messages(tokenizer, messages[:idx], add_generation_prompt=True)
        full_until_message = _render_messages(tokenizer, messages[: idx + 1], add_generation_prompt=False)

        if full_until_message.startswith(prefix_text):
            content_start = len(prefix_text)
            while content_start < len(full_until_message) and full_until_message[content_start].isspace():
                content_start += 1

            if full_until_message.startswith(content, content_start):
                content_end = content_start + len(content)
            else:
                matched_start = full_until_message.find(content, max(0, content_start - 32))
                if matched_start >= 0:
                    content_start = matched_start
                    content_end = matched_start + len(content)
                else:
                    content_end = len(full_until_message)
                    while content_end > content_start and full_until_message[content_end - 1].isspace():
                        content_end -= 1
        else:
            content_start = full_until_message.find(content)
            if content_start < 0:
                continue
            content_end = content_start + len(content)

        if content_end > content_start:
            spans.append((content_start, content_end))

    return spans


def _labels_from_char_spans(
    input_ids: Sequence[int],
    offset_mapping: Sequence[Tuple[int, int]],
    char_spans: Sequence[Tuple[int, int]],
) -> List[int]:
    labels = [-100] * len(input_ids)
    for token_idx, (token_start, token_end) in enumerate(offset_mapping):
        if token_end <= token_start:
            continue
        for span_start, span_end in char_spans:
            if token_start < span_end and token_end > span_start:
                labels[token_idx] = input_ids[token_idx]
                break
    return labels


def _labels_from_token_spans(
    tokenizer,
    messages: Sequence[Dict[str, str]],
    input_ids: Sequence[int],
) -> List[int]:
    labels = [-100] * len(input_ids)
    assistant_spans = _assistant_target_spans(tokenizer, messages)
    for start, end in assistant_spans:
        if start >= len(labels):
            continue
        bounded_end = min(end, len(labels))
        for idx in range(start, bounded_end):
            labels[idx] = input_ids[idx]
    return labels


def _tokenize_example(
    example: Dict[str, Any],
    tokenizer,
    max_length: int,
    system_prompt: str,
    dataset_name: str,
    dataset_format: str,
    messages_field: str,
    prepend_system_prompt: bool,
    label_mask_strategy: str,
) -> Dict[str, Any]:
    full_messages = _build_messages(
        example,
        dataset_name=dataset_name,
        system_prompt=system_prompt,
        dataset_format=dataset_format,
        messages_field=messages_field,
        prepend_system_prompt=prepend_system_prompt,
    )
    full_text = _render_messages(tokenizer, full_messages, add_generation_prompt=False)
    normalized_mask_strategy = str(label_mask_strategy or "token_length").lower()
    use_offset_mask = normalized_mask_strategy == "offset"
    if normalized_mask_strategy not in {"token_length", "offset"}:
        raise ValueError(
            f"Unsupported dataset.label_mask_strategy={label_mask_strategy!r}. "
            "Expected 'token_length' or 'offset'."
        )
    assistant_char_spans = _assistant_target_char_spans(tokenizer, full_messages) if use_offset_mask else []
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=bool(use_offset_mask and getattr(tokenizer, "is_fast", False)),
    )

    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]

    if not input_ids:
        return {"input_ids": [], "attention_mask": [], "labels": []}

    if "offset_mapping" in encoded and assistant_char_spans:
        labels = _labels_from_char_spans(input_ids, encoded["offset_mapping"], assistant_char_spans)
        if all(label == -100 for label in labels):
            labels = _labels_from_token_spans(tokenizer, full_messages, input_ids)
    else:
        labels = _labels_from_token_spans(tokenizer, full_messages, input_ids)

    if all(label == -100 for label in labels):
        return {"input_ids": [], "attention_mask": [], "labels": []}

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _load_raw_dataset(dataset_name: str, subset: str) -> DatasetDict:
    if subset:
        return load_dataset(dataset_name, subset)
    return load_dataset(dataset_name)


def _resolve_split(dataset: DatasetDict, preferred_name: str, fallback_names: Iterable[str]) -> Dataset:
    if preferred_name in dataset:
        return dataset[preferred_name]
    for name in fallback_names:
        if name in dataset:
            return dataset[name]
    raise KeyError(f"Could not resolve split {preferred_name!r}. Available: {list(dataset.keys())}")


def _matches_any_pattern(value: str, patterns: Sequence[str]) -> bool:
    normalized_value = value.lower()
    return any(pattern.lower() in normalized_value for pattern in patterns if pattern)


def _filter_by_source(dataset: Dataset, dataset_cfg: Dict[str, Any]) -> Dataset:
    source_field = dataset_cfg.get("source_field", "source")
    include_patterns = dataset_cfg.get("source_include_patterns") or []
    exclude_patterns = dataset_cfg.get("source_exclude_patterns") or []

    if not include_patterns and not exclude_patterns:
        return dataset
    if source_field not in dataset.column_names:
        raise KeyError(
            f"Configured source field {source_field!r} not found in dataset columns {dataset.column_names!r}."
        )

    def should_keep(example: Dict[str, Any]) -> bool:
        source_value = str(example.get(source_field, ""))
        if include_patterns and not _matches_any_pattern(source_value, include_patterns):
            return False
        if exclude_patterns and _matches_any_pattern(source_value, exclude_patterns):
            return False
        return True

    return dataset.filter(should_keep, desc="Filtering by source")


def _row_ids_to_index_map(row_ids: Sequence[int]) -> Dict[int, int]:
    return {int(row_id): idx for idx, row_id in enumerate(row_ids)}


def _select_rows_by_ids(dataset: Dataset, row_ids: Sequence[int], row_id_column: str) -> Dataset:
    id_to_idx = _row_ids_to_index_map(dataset[row_id_column])
    missing = [int(row_id) for row_id in row_ids if int(row_id) not in id_to_idx]
    if missing:
        raise KeyError(
            f"Cached split references {len(missing)} missing row ids. "
            "This usually means the source dataset or source filters changed."
        )
    return dataset.select([id_to_idx[int(row_id)] for row_id in row_ids])


def _normalize_cached_split_config(dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "dataset_name": dataset_cfg.get("name"),
        "subset": dataset_cfg.get("subset"),
        "source_field": dataset_cfg.get("source_field", "source"),
        "source_include_patterns": dataset_cfg.get("source_include_patterns") or [],
        "source_exclude_patterns": dataset_cfg.get("source_exclude_patterns") or [],
        "eval_holdout_size": dataset_cfg.get("eval_holdout_size"),
        "eval_holdout_ratio": dataset_cfg.get("eval_holdout_ratio", 0.01),
        "split_seed": dataset_cfg.get("split_seed", 42),
    }


def _build_eval_holdout(raw_train_dataset: Dataset, dataset_cfg: Dict[str, Any]) -> Tuple[Dataset, Dataset]:
    holdout_size = dataset_cfg.get("eval_holdout_size")
    holdout_ratio = dataset_cfg.get("eval_holdout_ratio", 0.01)
    split_seed = dataset_cfg.get("split_seed", 42)
    split_cache_path = dataset_cfg.get("split_cache_path")
    row_id_column = "_source_row_idx"

    if row_id_column not in raw_train_dataset.column_names:
        raw_train_dataset = raw_train_dataset.add_column(row_id_column, list(range(len(raw_train_dataset))))

    if split_cache_path:
        cache_path = Path(split_cache_path)
        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as handle:
                cached = json.load(handle)
            cached_cfg = cached.get("config", {})
            current_cfg = _normalize_cached_split_config(dataset_cfg)
            if cached_cfg != current_cfg:
                raise ValueError(
                    f"Split cache {cache_path} was created with a different dataset split config.\n"
                    f"Cached: {cached_cfg}\nCurrent: {current_cfg}"
                )
            train_dataset = _select_rows_by_ids(raw_train_dataset, cached["train_row_ids"], row_id_column)
            eval_dataset = _select_rows_by_ids(raw_train_dataset, cached["eval_row_ids"], row_id_column)
            return train_dataset, eval_dataset

    if holdout_size is None:
        test_size: Any = holdout_ratio
    else:
        test_size = min(int(holdout_size), max(len(raw_train_dataset) - 1, 1))

    split = raw_train_dataset.train_test_split(test_size=test_size, seed=split_seed, shuffle=True)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    if split_cache_path:
        cache_path = Path(split_cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": _normalize_cached_split_config(dataset_cfg),
            "train_row_ids": [int(row_id) for row_id in train_dataset[row_id_column]],
            "eval_row_ids": [int(row_id) for row_id in eval_dataset[row_id_column]],
        }
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    return train_dataset, eval_dataset


def _make_tokenize_fn(config: Dict[str, Any], tokenizer):
    dataset_cfg = config["dataset"]
    return lambda example: _tokenize_example(
        example,
        tokenizer=tokenizer,
        max_length=dataset_cfg["max_length"],
        system_prompt=dataset_cfg["system_prompt"],
        dataset_name=dataset_cfg["name"],
        dataset_format=dataset_cfg.get("format", "auto"),
        messages_field=dataset_cfg.get("messages_field", "messages"),
        prepend_system_prompt=dataset_cfg.get("prepend_system_prompt", False),
        label_mask_strategy=dataset_cfg.get("label_mask_strategy", "token_length"),
    )


def _tokenize_supervised_dataset(
    dataset: Dataset,
    tokenize_fn,
    num_proc: int,
    desc: str,
) -> Dataset:
    return dataset.map(
        tokenize_fn,
        remove_columns=list(dataset.column_names),
        num_proc=num_proc,
        desc=desc,
    ).filter(
        lambda ex: len(ex["input_ids"]) > 0,
        desc=f"Filtering empty {desc.lower()} samples",
    )


def prepare_datasets(config: Dict[str, Any], tokenizer) -> Tuple[Dataset, Dataset]:
    dataset_cfg = config["dataset"]
    raw = _load_raw_dataset(dataset_cfg["name"], dataset_cfg.get("subset"))

    train_dataset = _filter_by_source(_resolve_split(raw, dataset_cfg["train_split"], []), dataset_cfg)

    eval_split = dataset_cfg.get("eval_split")
    if eval_split:
        eval_dataset = _filter_by_source(
            _resolve_split(raw, eval_split, ["validation", "val", "dev", "test"]),
            dataset_cfg,
        )
    elif dataset_cfg.get("split_train_for_eval", False):
        train_dataset, eval_dataset = _build_eval_holdout(train_dataset, dataset_cfg)
    else:
        raise ValueError(
            "No eval split configured. Set dataset.eval_split or enable dataset.split_train_for_eval."
        )

    max_train_samples = dataset_cfg.get("max_train_samples")
    max_eval_samples = dataset_cfg.get("max_eval_samples")

    if max_train_samples:
        train_dataset = train_dataset.select(range(min(max_train_samples, len(train_dataset))))
    if max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    tokenize_fn = _make_tokenize_fn(config, tokenizer)
    num_proc = dataset_cfg.get("num_proc", 1)

    train_dataset = _tokenize_supervised_dataset(
        train_dataset,
        tokenize_fn,
        num_proc=num_proc,
        desc="Tokenizing train split",
    )
    eval_dataset = _tokenize_supervised_dataset(
        eval_dataset,
        tokenize_fn,
        num_proc=num_proc,
        desc="Tokenizing eval split",
    )

    return train_dataset, eval_dataset


def prepare_eval_dataset(config: Dict[str, Any], tokenizer) -> Dataset:
    dataset_cfg = config["dataset"]
    raw = _load_raw_dataset(dataset_cfg["name"], dataset_cfg.get("subset"))

    if dataset_cfg.get("eval_split"):
        eval_dataset = _filter_by_source(
            _resolve_split(raw, dataset_cfg["eval_split"], ["validation", "val", "dev", "test"]),
            dataset_cfg,
        )
    elif dataset_cfg.get("split_train_for_eval", False):
        train_dataset = _filter_by_source(_resolve_split(raw, dataset_cfg["train_split"], []), dataset_cfg)
        _, eval_dataset = _build_eval_holdout(train_dataset, dataset_cfg)
    else:
        raise ValueError(
            "No eval split configured. Set dataset.eval_split or enable dataset.split_train_for_eval."
        )

    max_eval_samples = dataset_cfg.get("max_eval_samples")
    if max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(max_eval_samples, len(eval_dataset))))

    return _tokenize_supervised_dataset(
        eval_dataset,
        _make_tokenize_fn(config, tokenizer),
        num_proc=dataset_cfg.get("num_proc", 1),
        desc="Tokenizing eval split",
    )
