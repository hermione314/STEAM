import os
import os.path as osp
import asyncio
from tqdm import tqdm
import torch
import re


class UserSimEvaluator:


    def __init__(
        self,
        host,
        expected_candidates: int = 20,
        topk_user_mem: int = 1,      
        topk_linked: int = 1,        
        threshold: float = 0.5,
    ):
        self.host = host
        self.expected_candidates = expected_candidates
        self.topk_user_mem = topk_user_mem
        self.topk_linked = topk_linked
        self.threshold = threshold

        
        self.sim_paths = {
            1: osp.join(host.data_path, "movie_sample100.sim.k1.inter"),
            3: osp.join(host.data_path, "movie_sample100.sim.k3.inter"),
            9: osp.join(host.data_path, "movie_sample100.sim.k9.inter"),
        }

        
        self._eval_done = False

   
    def _load_and_validate_sim_file(self, path: str, expected_candidates: int = 20):
        if not os.path.exists(path):
            raise FileNotFoundError(f"[UserSim] file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            
            expected_header = ["user_id:token", "init_item_list:token_seq", "candidate_item_list:token_seq", "label_list:token_seq"]
            if header[:4] != expected_header:
                raise ValueError(
                    f"[UserSim] Bad header in {path}.\n"
                    f"Got: {header}\n"
                    f"Expect startswith: {expected_header}"
                )

            data = []
            line_id = 1
            for line in f:
                line_id += 1
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    raise ValueError(f"[UserSim] Line {line_id} in {path} has <4 columns: {parts}")

                user_str = parts[0].strip()
                init_str = parts[1].strip()
                cand_str = parts[2].strip()
                label_str = parts[3].strip()

                if user_str == "":
                    raise ValueError(f"[UserSim] Line {line_id} empty user_id in {path}")

                try:
                    user_id = int(user_str)
                except Exception:
                    raise ValueError(f"[UserSim] Line {line_id} user_id not int: '{user_str}' in {path}")

                init_items = [int(x) for x in init_str.split()] if init_str else []
                cand_items = [int(x) for x in cand_str.split()] if cand_str else []
                labels = [int(x) for x in label_str.split()] if label_str else []

                if len(cand_items) != expected_candidates:
                    raise ValueError(
                        f"[UserSim] Line {line_id} candidates length != {expected_candidates} in {path}. "
                        f"Got {len(cand_items)}"
                    )
                if len(labels) != expected_candidates:
                    raise ValueError(
                        f"[UserSim] Line {line_id} labels length != {expected_candidates} in {path}. "
                        f"Got {len(labels)}"
                    )
                for y in labels:
                    if y not in (0, 1):
                        raise ValueError(f"[UserSim] Line {line_id} label not 0/1 in {path}: {labels}")

                data.append(
                    {
                        "user_id": user_id,
                        "init_items": init_items,
                        "candidates": cand_items,
                        "labels": labels,
                    }
                )

        if len(data) == 0:
            raise ValueError(f"[UserSim] No data rows found in {path}")

        print(f"[UserSim] Loaded {len(data)} rows from {path}")
        return data

    
    def _get_item_emb_by_raw_id(self, raw_item_id: int):
        
        host = self.host
        key = str(int(raw_item_id))
        if key in host.item_id_to_index:
            idx = host.item_id_to_index[key]
        else:
            
            idx = host.item_id_list.index(key)
        return host.item_embeddings[idx]

    def _retrieve_one_user_memory(self, user_id: int, cand_item: int):
        
        host = self.host
        if user_id not in host.user_agents:
            return None, ""

        ua = host.user_agents[user_id]
        if (not hasattr(ua, "memory_emb")) or ua.memory_emb is None or len(ua.memory_emb) == 0:
            return None, ""

        cand_emb = self._get_item_emb_by_raw_id(cand_item).to(host.device)
        mem_embs = [m.to(host.device) for m in ua.memory_emb]

        mem_stack = torch.stack(mem_embs, dim=0)
        sims = torch.matmul(cand_emb, mem_stack.T)
        sims = (sims + 1) / 2
        sims = torch.where(sims == 1, torch.tensor(float("-inf"), device=sims.device), sims)

        topv, topi = torch.topk(sims, k=1)
        mid = int(topi[0].item())
        try:
            mem_text = str(ua.memory_1[mid])
        except Exception:
            mem_text = ""
        return mid, mem_text

    def _retrieve_one_linked_info(self, user_id: int, mem_idx: int, cand_item: int):
       
        host = self.host
        if user_id not in host.user_agents:
            return ""

        ua = host.user_agents[user_id]
        if not isinstance(getattr(ua, "link", None), dict):
            return ""

        if mem_idx is None or mem_idx not in ua.link or len(ua.link[mem_idx]) == 0:
            return ""

        cand_emb = self._get_item_emb_by_raw_id(cand_item).to(host.device)

       
        try:
            cluster_id = host.get_linked_cluster(ua.link[mem_idx], [cand_emb])[0]
        except Exception:
            cluster_id = ua.link[mem_idx][0]

        if cluster_id not in host.clustered_memory:
            return ""

        cluster_link = list(host.clustered_memory[cluster_id].get("link", []))
        if len(cluster_link) == 0:
            return ""

        linked_embs = []
        linked_gidxs = []
        for gidx in cluster_link:
            if gidx < 0 or gidx >= len(host.global_memory_info):
                continue
            uid, mid = host.global_memory_info[gidx]
            if uid == user_id:
                continue
            linked_embs.append(host.global_memory_embs[gidx].to(host.device))
            linked_gidxs.append(gidx)

        if len(linked_embs) == 0:
            return ""

        linked_stack = torch.stack(linked_embs, dim=0)
        sims = torch.matmul(cand_emb, linked_stack.T)
        sims = (sims + 1) / 2
        sims = torch.where(sims == 1, torch.tensor(float("-inf"), device=sims.device), sims)
        topv, topi = torch.topk(sims, k=1)

        best_pos = int(topi[0].item())
        best_gidx = linked_gidxs[best_pos]
        uid, mid = host.global_memory_info[best_gidx]

        try:
            mem_txt = str(host.user_agents[uid].memory_1[mid])
        except Exception:
            mem_txt = ""

        inter_titles = []
        try:
            inter_items = host.user_agents[uid].intered_items[mid]
            for it in inter_items[:5]:
                inter_titles.append(str(it))
        except Exception:
            pass

        return f"[RelatedUserMemory]\nMemory: {mem_txt}\nInteracted movies: {', '.join(inter_titles) if inter_titles else '[None]'}"

    
    def build_batch_prompt(self, user_id: int, init_items: list, candidates: list):
        """
        一次prompt判断20个候选，输出20个0/1
        """
        host = self.host
        user_desc = host.user_agents[user_id].memory_1[-1] if user_id in host.user_agents else "[PAD]"

        his_titles = []
        for it in init_items[-20:]:
            try:
                his_titles.append(host.item_text[int(it)])
            except Exception:
                his_titles.append(str(it))
        user_his_text = "\n".join([f"{i+1}. {t}" for i, t in enumerate(his_titles)]) if his_titles else "[None]"

        cand_blocks = []
        for j, cand in enumerate(candidates):
            try:
                cand_title = host.item_text[int(cand)]
            except Exception:
                cand_title = str(cand)

            cand_desc = host.item_descriptions.get(str(int(cand)), "")

            mem_idx, mem_text = self._retrieve_one_user_memory(user_id, cand)
            linked_info = self._retrieve_one_linked_info(user_id, mem_idx, cand)

            if mem_text == "":
                mem_text = "[No matching user memory retrieved]"
            if linked_info == "":
                linked_info = "[No related user memory retrieved]"

            block = f"""
Candidate {j+1}:
- item_id: {int(cand)}
- Title: {cand_title}
- Description: {cand_desc}

Most relevant user memory (top1):
{mem_text}

Related user memory (top1):
{linked_info}
""".strip()
            cand_blocks.append(block)

        cand_section = "\n\n".join(cand_blocks)

        prompt = f"""
You are a movie shopper on Amazon.com. Your task is to judge, for EACH candidate movie below, whether you would LIKE it.

Your self-description:
{user_desc}

Your recent interactions (history):
{user_his_text}

Now evaluate the following candidates:
{cand_section}

Decision requirements:
1) For each candidate, retrieve your preferences from the provided "Most relevant user memory" and your self-description.
2) Use the "Related user memory" only as weak supporting evidence (homogeneity).
3) If preferences are unclear, use common-sense signals (general appeal) to decide.
4) Output a binary decision for EACH candidate.

Output format (STRICT):
Return ONLY one line containing EXACTLY 20 integers separated by spaces.
- The i-th integer corresponds to Candidate i.
- Use 1 if you LIKE it, 0 if you DISLIKE it.

Answer:
""".strip()

        return prompt

    def parse_batch_output(self, text: str, expected_len: int = 20):
        nums = re.findall(r"\b[01]\b", text)
        if len(nums) < expected_len:
            raise ValueError(f"[UserSim] Cannot parse {expected_len} binary labels. Got {len(nums)}. Raw: {text[:200]}")
        nums = nums[:expected_len]
        return [int(x) for x in nums]

   
    def _compute_prf(self, y_true, y_pred):
        tp = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 1)
        fp = sum(1 for a, b in zip(y_true, y_pred) if a == 0 and b == 1)
        fn = sum(1 for a, b in zip(y_true, y_pred) if a == 1 and b == 0)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        return precision, recall, f1

   
    def evaluate_all_k(self, force: bool = False):
        
        if self._eval_done and (not force):
            return None

        host = self.host
        results = {}

        for k, data in self.sim_data.items():
            prompts = []
            labels_list = []
            for row in data:
                user_id = row["user_id"]
                init_items = row["init_items"]
                candidates = row["candidates"]
                labels = row["labels"]

                prompts.append(self.build_batch_prompt(user_id, init_items, candidates))
                labels_list.append(labels)

            messages = []
            for i in tqdm(range(0, len(prompts), host.chat_api_batch), desc=f"[UserSim] LLM batch k={k}"):
                chunk = prompts[i:i + host.chat_api_batch]
                
                messages += asyncio.run(host.user_agents[0].llm_chat.agenerate_response_without_construction(chunk))

            all_true = []
            all_pred = []
            for idx, resp in enumerate(messages):
                content = resp["choices"][0]["message"]["content"]
                pred = self.parse_batch_output(content, expected_len=self.expected_candidates)
                true = labels_list[idx]
                all_true.extend(true)
                all_pred.extend(pred)

            precision, recall, f1 = self._compute_prf(all_true, all_pred)
            results[k] = (precision, recall, f1)
            print(f"[UserSim][k={k}] Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}")

        self._eval_done = True
        return results
