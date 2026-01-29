# model/item_preprocess.py
import os
import json
import time
from typing import Dict, List, Optional, Tuple

import openai
from tqdm import tqdm
from openai import error as openai_error


class MovieDescriptionProcessor:
    def __init__(
        self,
        data_path: str,
        dataset_name: str,
        item_text: List[str],
        item_token_id: Dict[str, int],
        *,
        model: str = "gpt-3.5-turbo",
        api_key: Optional[str] = 'sk-wtdMxBvZ3eszhiBMpQmrgjbWgmWENgT8tVqniyHfPcQ3VJyq',
        out_filename: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 200,
        show_progress: bool = True,
        retry_times: int = 1,           # 新增：单条最多重试次数
        sleep_on_error: float = 1.0,    # 新增：出错后等待时间（秒）
    ) -> None:
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.item_text = item_text
        self.item_token_id = item_token_id
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.show_progress = show_progress
        self.retry_times = retry_times
        self.sleep_on_error = sleep_on_error

        self.out_path = os.path.join(
            data_path,
            out_filename or f"{dataset_name}.item_descriptions.json",
        )

        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("请设置 OPENAI_API_KEY 或在构造器传入 api_key。")
        openai.api_key = key

    # -----------------------------
    # 主入口（同步 + 容错）
    # -----------------------------
    def process_items(self, feat_path: str) -> Dict[int, str]:
        items = self._parse_feat_file(feat_path)
        out: Dict[int, str] = {}

        iterator = items
        if self.show_progress:
            iterator = tqdm(
                items,
                total=len(items),
                desc=f"Generating item descriptions ({self.dataset_name})",
                unit="item",
                dynamic_ncols=True,
                leave=False,
            )

        for _, token_id, title, category, release_year, brand, sales_type in iterator:
            desc = self._describe_once_with_fallback(title, category, brand)
            out[token_id] = desc

        # 写盘
        os.makedirs(os.path.dirname(self.out_path), exist_ok=True)
        with open(self.out_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in out.items()}, f, ensure_ascii=False, indent=2)

        print(f"\n所有描述已保存至：{self.out_path}")
        return out

    # -----------------------------
    # 带重试 + 兜底的描述生成
    # -----------------------------
    def _describe_once_with_fallback(
        self, title: str, category: Optional[str], release_year: Optional[str]
    ) -> str:
        for attempt in range(1, self.retry_times + 2):  # +1 是为了最后一次直接走兜底
            try:
                return self._describe_once(title, category, release_year)
            except openai_error.InvalidRequestError as e:
                if "content management policy" in str(e) or "filtered" in str(e).lower():
                    print(f"\n[Azure 内容过滤] 跳过敏感商品 → Title: {title}")
                    return title  # 直接返回标题作为描述
                else:
                    print(f"\n[InvalidRequestError] {title} | 尝试 {attempt}/{self.retry_times} | {e}")
            except openai_error.RateLimitError:
                print(f"\n[RateLimit] 等待 {self.sleep_on_error}s 后重试 → {title}")
                time.sleep(self.sleep_on_error * 2)
            except (openai_error.Timeout, openai_error.APIConnectionError):
                print(f"\n[网络超时/连接错误] 尝试 {attempt}/{self.retry_times} → {title}")
                if attempt <= self.retry_times:
                    time.sleep(self.sleep_on_error)
            except openai_error.APIError as e:
                print(f"\n[OpenAI API 错误] {e} | Title: {title}")

            if attempt <= self.retry_times:
                time.sleep(self.sleep_on_error)
            else:
                # 所有重试都失败了，用 title 兜底
                print(f"[最终兜底] 使用原始标题 → {title}")
                return title

        return title  # 保险起见再 return 一次

    # -----------------------------
    # 原版 _describe_once（保持不变，只是被上面包装了一层）
    # -----------------------------
    def _describe_once(
        self, title: str, category: Optional[str], release_year: Optional[str]
    ) -> str:
        if "ml-" in self.dataset_name or self.dataset_name.startswith("ml-"):
            sys_prompt = """
            You are a professional film description assistant.
            Your task is to write professional, formal, and informative film descriptions.
            You must provide the title and categorization of each film.
            Your description should focus on:
            - Basic film information, such as the director, main cast, running time, and release year (if obvious or inferred);
            - A brief description of the plot or theme.
            Please begin your response with the film's title. Please keep your response concise (80 words or less).
            Below is an example of a high-quality response:
            "Home Free" is a 2023 American comedy film directed by Aaron Brown and written by Lenny Barszap. Set in 1997, the 98-minute film tells the story of a group of college students who, despite their parents' objections, invite a homeless professor to stay on his porch. The film explores themes of community, acceptance, and friendship. Furthermore, the film inspired the "Been There" movement, which promotes social change through entertainment, specifically addressing homelessness.
            """
        elif "CD" in self.dataset_name or self.dataset_name.startswith("CD"):
            sys_prompt = """
            You are a professional CD description assistant.
            Your task is to write professional, formal, and informative CD descriptions.
            You must provide the title and category of each CD.
            Your description should focus on:
            - The CD's artist, release date, whether it's a film soundtrack (if not, don't mention it), and any awards it has won (if not, don't mention it).
            - The CD's genre and background.
            Please begin your response with the CD title and keep it concise (80 characters or less).
            The following is an example of a high-quality response:
            **Make It Last Forever** Released in 1987, *Make It Last Forever* is the debut album by Keith Sweat, a pivotal figure in contemporary R\&B and the New Jack Swing movement. The record’s smooth production and heartfelt vocals established Sweat as a defining voice of the era, featuring enduring hits such as “I Want Her.” Its blend of soulful melodies and modern rhythms helped shape late-1980s R\&B, earning widespread acclaim and solidifying its place as a genre classic.
            """

        else:
            sys_prompt = """
            You are a professional clothing recommendation expert. Your task is to generate compelling product descriptions based on the Amazon clothing dataset to help users discover suitable clothing.
            You must provide the following for each clothing item: title, categories, brand (use 'nan' if unknown), and sales type.
            Please generate a concise and professional description (within 80 characters), highlighting the following aspects to optimize recommendations:
            - Core Features: Extract style, material, purpose, and unique design (e.g., closure, pockets) from the title and categories.
            - Applicable Scenarios: Infer the occasion based on categories (e.g., casual, sporty, everyday).
            - Brand and Quality: If the brand is not nan, emphasize its reputation; combine with the sales type to highlight comfort or durability.
            - Recommendation Attractiveness: Describe why it suits the user in a positive way (e.g., comfortable, stylish, versatile), avoiding generalization and focusing on improving user fit.
            The output should begin with the clothing item's title and consist only of descriptive text; do not add any additional explanations.
            Here is an example of a high-quality description: **The Heavy Blend Adult Hooded Sweatshirt** is designed for men's everyday casual and athletic occasions. Made from high-quality fabric, it ensures softness, comfort, and durability, maintaining its shape even during active activities. The classic hooded design features a drawstring closure for easy adjustment, and a large front pocket provides convenient storage for small items. Perfect for the gym, outdoor walks, or a relaxing weekend. Whether you're looking for a versatile sweatshirt or everyday warmth, this affordable, brand-agnostic option offers reliable comfort and style, helping you easily pair it with jeans or sweatpants for an energetic look.
            """

        user_prompt = self._format_user_prompt(title, category, release_year)

        resp = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": sys_prompt.strip()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = resp["choices"][0]["message"]["content"] or ""
        return content.strip()

    # -----------------------------
    # 其余函数完全不变
    # -----------------------------
    def _format_user_prompt(
        self, title: str, category: Optional[str], brand: Optional[str]
    ) -> str:
        parts = [f"Title: {title}"]
        if category:
            parts.append(f"Category/Genre: {category}")
        if brand:
            parts.append(f"Release Year: {brand}")
        return "\n".join(parts)

    # 解析 feat 文件
    # -----------------------------
    def _parse_feat_file(
        self, feat_path: str
    ) -> List[Tuple[str, int, str, Optional[str], Optional[str]]]:
        """
        Returns: List[(item_id_str, token_id, title, category/genre, release_year)]
        """
        results: List[Tuple[str, int, str, Optional[str], Optional[str]]] = []

        with open(feat_path, "r", encoding="utf-8") as f:
            _ = f.readline()  # 跳过表头

            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split("\t")
                item_id_str: Optional[str] = None
                title: Optional[str] = None
                category: Optional[str] = None
                year: Optional[str] = None
                brand: Optional[str] = None
                sales_type: Optional[str] = None

                # 兼容 ml-* 与 CDs 两种格式
                if "ml-" in self.dataset_name or self.dataset_name.startswith("ml-"):
                    # ml-*：item_id, movie_title, release_year, genre
                    item_id_str = parts[0] if len(parts) > 0 else None
                    title = parts[1] if len(parts) > 1 else None
                    year = parts[2] if len(parts) > 2 else None
                    category = parts[3] if len(parts) > 3 else None
                elif "CD" in self.dataset_name or self.dataset_name.startswith("CD"):
                    # CDs：item_id, item_title[, item_class]
                    item_id_str = parts[0] if len(parts) > 0 else None
                    title = parts[1] if len(parts) > 1 else None
                    category = parts[2] if len(parts) > 2 else "CDs"
                else:
                    item_id_str = parts[0] if len(parts) > 0 else None
                    title = parts[1] if len(parts) > 1 else None
                    category = parts[2] if len(parts) > 2 else None
                    brand = parts[3] if len(parts) > 3 else None
                    sales_type= parts[4] if len(parts) > 4 else None

                if not item_id_str or item_id_str not in self.item_token_id:
                    continue

                token_id = self.item_token_id[item_id_str]

                # 优先使用 item_text[token_id]（你的 load_text 已做清洗）
                final_title = self._safe_item_title(token_id) or title or "Unknown Title"

                results.append((item_id_str, token_id, final_title, category, year, brand, sales_type))

        return results

    def _safe_item_title(self, token_id: int) -> Optional[str]:
        if 0 <= token_id < len(self.item_text):
            return self.item_text[token_id]
        return None

