import pickle
import asyncio
import torch
from logging import getLogger
from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.utils import InputType
import os.path as osp
import os
import glob
import json
from agentverse.initialization import load_agent
from fuzzywuzzy import process
from tqdm import tqdm
from openai.embeddings_utils import (
    get_embedding,
    distances_from_embeddings,
    indices_of_nearest_neighbors_from_distances,
)
from model.item_preprocess import MovieDescriptionProcessor
from transformers import AutoTokenizer, AutoModel
import time
from collections import deque
import math


class STEAM(SequentialRecommender):
    """LLM-based sequential recommender using user memory and collaborative filtering links."""
    
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(STEAM, self).__init__(config, dataset)
        self.n_users = dataset.num(self.USER_ID)
        self.config = config
        self.sample_num = config['sample_num']
        self.embedding_size = config["embedding_size"]
        self.data_path = config['data_path']
        self.dataset_name = dataset.dataset_name
        self.logger = getLogger()
        self.item_token_id = dataset.field2token_id['item_id']
        self.item_id_token = dataset.field2id_token['item_id']
        self.user_id_token = dataset.field2id_token['user_id']
        self.user_token_id = dataset.field2token_id['user_id']
        self.apply(xavier_normal_initialization)

        self.api_batch = config['api_batch']
        self.chat_api_batch = config['chat_api_batch']

        embedding_context = {
            'agent_type': 'embeddingagent', 'role_task': '', 'memory': [], 'prompt_template': '',
            'llm': {'model': self.config['embedding_model'], 'temperature': self.config['llm_temperature'],
                    'max_tokens': self.config['max_tokens'], 'llm_type': 'embedding',
                    'api_key_list': self.config['api_key_list'], 'current_key_idx': self.config['current_key_idx']},
            'llm_chat': {'model': 'gpt-3.5-turbo-16k-0613', 'llm_type': 'gpt-3.5-turbo-16k-0613',
                         'temperature': self.config['llm_temperature'], 'max_tokens': self.config['max_tokens_chat'],
                         'api_key_list': self.config['api_key_list'], 'current_key_idx': self.config['current_key_idx']},
            'agent_mode': 'embedding', 'output_parser_type': 'recommender',
        }
        self.embedding_agent = load_agent(embedding_context)

        self.item_text = self.load_text()
        self.user_context = self.load_user_context()

        self.record_idx = 0
        self.global_memory_embs = []
        self.global_memory_info = []
        self.clustered_memory = {}
        self.max_his_len = 20

        self.forward_retrieval_k = config['forward_retrieval_k']
        self.backward_retrieval_k = config['backward_retrieval_k']
        self.link_k1 = config['link_k1']
        self.link_threshold = config['link_threshold']
        self.max_hop = config['max_hop']
        self.max_component_size = config['max_component_size']

        _LOCAL_MODEL_DIR = "hugginface_model/all-MiniLM-L6-v2"
        self._model = AutoModel.from_pretrained(_LOCAL_MODEL_DIR, local_files_only=True).to(self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(_LOCAL_MODEL_DIR, local_files_only=True)
        self._model.eval()

        while True:
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
            if os.path.exists(path):
                self.record_idx += 1
            else:
                break

        print(f"Record path: user_record_{self.record_idx}")
        print(f"forward_k: {self.forward_retrieval_k}, backward_k: {self.backward_retrieval_k}, link_k1: {self.link_k1}, max_hop: {self.max_hop}, max_comp_size: {self.max_component_size}")

        self.user_agents = {}
        for user_id, user_context in self.user_context.items():
            agent = load_agent(user_context)
            self.user_agents[user_id] = agent
            path = osp.join(self.config['record_path'], self.dataset_name, 'record', f'user_record_{self.record_idx}')
            if not os.path.exists(path):
                os.makedirs(path)
            with open(osp.join(path, f'user.{str(user_id)}'), 'w') as f:
                f.write('~' * 20 + 'Meta information' + '~' * 20 + '\n')
                f.write(f'The user wrote the following self-description as follows: {user_context["update_memory"][-1]}\n')

        rec_context = {
            'agent_type': 'recagent', 'memory': [], 'prompt_template': self.config['system_prompt_template'],
            'llm': {'model': self.config['llm_model'], 'llm_type': self.config['llm_model'],
                    'temperature': self.config['llm_temperature'], 'max_tokens': self.config['max_tokens'],
                    'api_key_list': self.config['api_key_list'], 'current_key_idx': self.config['current_key_idx']},
            'llm_chat': {'model': 'gpt-3.5-turbo-16k-0613', 'llm_type': 'gpt-3.5-turbo-16k-0613',
                         'temperature': self.config['llm_temperature_test'], 'max_tokens': self.config['max_tokens_chat'],
                         'api_key_list': self.config['api_key_list'], 'current_key_idx': self.config['current_key_idx']},
            'agent_mode': 'system', 'output_parser_type': 'recommender',
            'system_prompt_template_backward': self.config['system_prompt_template_backward'],
            'system_prompt_template_evaluation_basic': self.config['system_prompt_template_evaluation_basic'],
            'system_prompt_template_evaluation_sequential': self.config['system_prompt_template_evaluation_sequential'],
            'system_prompt_template_evaluation_retrieval': self.config['system_prompt_template_evaluation_retrieval'],
            'prompt_template_linked': self.config['prompt_template_linked'],
            'prompt_template_clustered': self.config['prompt_template_clustered'],
            'n_users': self.n_users
        }
        self.rec_agent = load_agent(rec_context)

        feat_path = osp.join(self.data_path, f'movie.item')
        json_files = glob.glob(os.path.join(self.data_path, "*.json"))
        if json_files:
            with open(json_files[0], "r", encoding="utf-8") as f:
                self.item_descriptions = json.load(f)
        else:
            processor = MovieDescriptionProcessor(self.data_path, self.dataset_name, self.item_text, self.item_token_id)
            self.item_descriptions = processor.process_items(feat_path)

        id_files = glob.glob(os.path.join(self.data_path, "*.pkl"))
        emb_files = glob.glob(os.path.join(self.data_path, "*.pt"))
        if id_files and emb_files:
            self.item_id_list = sorted(self.item_descriptions.keys())
            self.item_embeddings = torch.load(emb_files[0], map_location=torch.device('cpu'))
        else:
            self.get_item_embs()
        self.item_id_to_index = {item_id: i for i, item_id in enumerate(self.item_id_list)}

        self.item_categories = self.load_item_categories()
        self.diversity_metrics = {'ild_list': [], 'entropy_list': [], 'ds_list': []}

    def load_item_categories(self):
        """Load category info from item file and build internal index to category mapping."""
        item_categories = {}
        
        if 'movie' in self.dataset_name:
            feat_path = osp.join(self.data_path, 'movie.item')
        elif 'cd' in self.dataset_name:
            feat_path = osp.join(self.data_path, 'cd.item')
        else:
            feat_path = osp.join(self.data_path, 'clothings.item')
        
        try:
            token_to_category = {}
            
            with open(feat_path, 'r', encoding='utf-8') as file:
                header = file.readline().strip().split('\t')
                
                category_idx = None
                for idx, col_name in enumerate(header):
                    col_lower = col_name.lower()
                    if 'class' in col_lower or 'categor' in col_lower or 'genre' in col_lower:
                        category_idx = idx
                        break
                
                if category_idx is None:
                    print(f"Warning: Cannot find category column. Headers: {header}")
                    return item_categories
                
                item_id_idx = 0
                for idx, col_name in enumerate(header):
                    if 'item_id' in col_name.lower():
                        item_id_idx = idx
                        break
                
                for line in file:
                    parts = line.strip().split('\t')
                    if len(parts) > max(category_idx, item_id_idx):
                        file_item_id = parts[item_id_idx].strip()
                        category = parts[category_idx].strip()
                        if file_item_id and category:
                            token_to_category[file_item_id] = category
            
            print(f"Loaded {len(token_to_category)} item categories from file")
            
            for internal_idx in range(len(self.item_id_token)):
                token = self.item_id_token[internal_idx]
                if token == '[PAD]':
                    continue
                
                category = None
                if token in token_to_category:
                    category = token_to_category[token]
                elif str(internal_idx) in token_to_category:
                    category = token_to_category[str(internal_idx)]
                elif str(internal_idx - 1) in token_to_category:
                    category = token_to_category[str(internal_idx - 1)]
                
                if category:
                    item_categories[internal_idx] = category
            
            print(f"Built {len(item_categories)} internal index to category mappings")
            if item_categories:
                sample_keys = list(item_categories.keys())[:3]
                print(f"Sample: {[(k, item_categories[k][:50]) for k in sample_keys]}")
                        
        except Exception as e:
            print(f"Warning: Error loading item categories: {e}")
            import traceback
            traceback.print_exc()
        
        return item_categories

    def _get_item_embedding_by_id(self, item_id):
        item_id_str = str(int(item_id))
        return self.item_embeddings[self.item_id_to_index[item_id_str]] if item_id_str in self.item_id_to_index else None
    
    def _get_item_categories_by_id(self, item_id):
        item_id_int = int(item_id)
        categories_str = self.item_categories.get(item_id_int)
        
        if categories_str is None:
            return []
        
        if '|' in categories_str:
            return [c.strip() for c in categories_str.split('|') if c.strip()]
        elif ',' in categories_str:
            return [c.strip() for c in categories_str.split(',') if c.strip()]
        else:
            return categories_str.split()
    
    def compute_ild(self, item_ids, top_k=5):
        """Compute Intra-List Diversity: average pairwise Euclidean distance of item embeddings."""
        item_ids = item_ids[:top_k]
        embeddings = [e for e in (self._get_item_embedding_by_id(i) for i in item_ids) if e is not None]
        n = len(embeddings)
        if n <= 1:
            return 0.0
        embeddings_tensor = torch.stack(embeddings)
        total_distance = sum(torch.norm(embeddings_tensor[i] - embeddings_tensor[j], p=2).item() 
                            for i in range(n) for j in range(i + 1, n))
        return (2.0 * total_distance) / (n * (n - 1))
    
    def compute_entropy(self, item_ids, top_k=5):
        """Compute category entropy: measures uniformity of category distribution."""
        item_ids = item_ids[:top_k]
        if not item_ids:
            return 0.0
        category_counts = {}
        for item_id in item_ids:
            for cat in self._get_item_categories_by_id(item_id):
                category_counts[cat] = category_counts.get(cat, 0) + 1
        if not category_counts:
            return 0.0
        total = sum(category_counts.values())
        return -sum((c/total) * math.log(c/total) for c in category_counts.values() if c > 0)
    
    def compute_diversity_score(self, item_ids, top_k=5):
        """Compute diversity score: ratio of unique categories to number of items."""
        item_ids = item_ids[:top_k]
        if not item_ids:
            return 0.0
        unique_cats = set()
        for item_id in item_ids:
            unique_cats.update(self._get_item_categories_by_id(item_id))
        return len(unique_cats) / len(item_ids)
    
    def reset_diversity_metrics(self):
        self.diversity_metrics = {'ild_list': [], 'entropy_list': [], 'ds_list': []}
    
    def print_diversity_results(self):
        results = {k: sum(self.diversity_metrics[f'{k}_list'])/len(self.diversity_metrics[f'{k}_list']) 
                   if self.diversity_metrics[f'{k}_list'] else 0.0 for k in ['ild', 'entropy', 'ds']}
        print(f"\n{'='*50}\nDiversity Metrics (Top-5)\n{'='*50}")
        print(f"ILD: {results['ild']:.4f}\nEntropy: {results['entropy']:.4f}\nDS: {results['ds']:.4f}")
        print(f"Samples: {len(self.diversity_metrics['ild_list'])}\n{'='*50}\n")
        return results

    def get_item_embs(self):
        """Generate embeddings for all items and save to disk."""
        item_ids = sorted(self.item_descriptions.keys())
        embeddings = [self.generate_embedding(self.item_descriptions[item_id]) for item_id in tqdm(item_ids, desc="Processing item embeddings")]
        self.item_id_list = item_ids
        self.item_embeddings = torch.stack(embeddings)
        with open(osp.join(self.data_path, 'item_id_list.pkl'), "wb") as f:
            pickle.dump(self.item_id_list, f)
        torch.save(self.item_embeddings, osp.join(self.data_path, 'item_embs.pt'))

    def load_user_context(self):
        user_context = {}
        base = {'agent_type':'useragent', 'role_description':{'age': '[PAD]', 'user_gender': '[PAD]','user_occupation':'[PAD]'},'update_memory':['[PAD]'],
                'role_description_string_1':'[PAD]','role_description_string_3':'[PAD]', 'role_task':'[PAD]','prompt_template': self.config['user_prompt_template'], 
                'user_prompt_system_role': self.config['user_prompt_system_role'],
                'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                'llm_chat':{'model':'gpt-3.5-turbo-16k-0613','llm_type':'gpt-3.5-turbo-16k-0613','temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                'agent_mode':'user','output_parser_type':'useragent','historical_interactions':[], 'user_prompt_template_true': self.config['user_prompt_template_true'], 
                'user_prompt_template_consolid': self.config['user_prompt_template_consolid'],'user_prompt_template_update': self.config['user_prompt_template_update'], 
                'user_prompt_template_inference': self.config['user_prompt_template_inference'],'memory_1':[],'memory_emb':[],'intered_items':[], 'link':[]}
        user_context[0] = base
        for user_id in range(self.n_users):
            user_context[user_id] = {'agent_type':'useragent','role_description': {},'role_description_string_3': f'This user enjoys listening CDs very much.',
                'role_description_string_1': f'I enjoy listening to CDs very much.','user_prompt_system_role': self.config['user_prompt_system_role'],
                'update_memory': [f' I enjoy listening to CDs very much.'],'prompt_template': self.config['user_prompt_template'],
                'llm':{'model':self.config['llm_model'],'llm_type':self.config['llm_model'],'temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                'llm_chat':{'model':'gpt-3.5-turbo-16k-0613','llm_type':'gpt-3.5-turbo-16k-0613','temperature':self.config['llm_temperature'],'max_tokens':self.config['max_tokens_chat'],'api_key_list':self.config['api_key_list'],'current_key_idx': self.config['current_key_idx']},
                'agent_mode':'user','output_parser_type':'useragent','historical_interactions':[], 'user_prompt_template_true': self.config['user_prompt_template_true'], 
                'user_prompt_template_consolid': self.config['user_prompt_template_consolid'],'user_prompt_template_update': self.config['user_prompt_template_update'],
                'user_prompt_template_inference': self.config['user_prompt_template_inference'],'memory_1':[],'memory_emb':[],'intered_items':[],'link':[]}
        return user_context

    def load_text(self):
        token_text = {}
        item_text = ['[PAD]']
        if 'movie' in self.dataset_name:
            feat_path = osp.join(self.data_path, 'movie.item')
        elif 'cd' in self.dataset_name:
            feat_path = osp.join(self.data_path, 'cd.item')
        else:
            feat_path = osp.join(self.data_path, 'clothings.item')
        with open(feat_path, 'r', encoding='utf-8') as file:
            file.readline()
            for line in file:
                parts = line.strip().split('\t')
                token_text[parts[0]] = parts[1]
        for token in self.item_id_token:
            if token == '[PAD]':
                continue
            raw_text = token_text.get(token, token)
            if raw_text.endswith(', The'):
                raw_text = 'The ' + raw_text[:-5]
            elif raw_text.endswith(', A'):
                raw_text = 'A ' + raw_text[:-3]
            item_text.append(raw_text)
        return item_text

    def generate_embedding(self, text):
        """Generate normalized embedding vector for input text using local model."""
        if not isinstance(text, str):
            text = str(text)
        with torch.no_grad():
            inputs = self._tokenizer(text, return_tensors="pt", truncation=True, max_length=256, padding=False)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            hidden = outputs.last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1)
            emb = (torch.sum(hidden * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)).squeeze(0)
            emb = emb / (emb.norm(p=2) + 1e-12)
        return emb

    def forward_retrieval(self, item_emb, memory_embs, k=1, threshold=0.5):
        """Retrieve top-k most similar memories based on cosine similarity."""
        if len(memory_embs) == 0:
            return [], []
        memory_embs_stack = torch.stack(memory_embs)
        sims = torch.matmul(item_emb, memory_embs_stack.T)
        sims = (sims + 1) / 2
        sims = torch.where(sims == 1, torch.tensor(float('-inf'), device=sims.device), sims)
        k = min(len(memory_embs), k)
        top_k = torch.topk(sims, k=k)
        keep_mask = top_k.values > threshold
        return top_k.indices[keep_mask].tolist(), top_k.values[keep_mask].tolist()

    def replace_memory_emb(self, user_id, memory_id, new_emb):
        for idx, (uid, mid) in enumerate(self.global_memory_info):
            if uid == user_id and mid == memory_id:
                self.global_memory_embs[idx] = new_emb
                return

    def get_global_idx(self, user_id, mem_idx):
        try:
            return self.global_memory_info.index((user_id, mem_idx))
        except ValueError:
            return None

    def find_cf_links(self, emb, user_id):
        """Find similar memories from other users for collaborative filtering."""
        if len(self.global_memory_embs) == 0:
            return []
        indices, _ = self.forward_retrieval(emb, self.global_memory_embs, k=self.link_k1, threshold=self.link_threshold)
        return [idx for idx in indices if self.global_memory_info[idx][0] != user_id][:self.link_k1]

    def _bfs_limited_component(self, start_node, adj, max_hop, max_size=8):
        visited = {start_node}
        queue = deque([(start_node, 0)])
        while queue and len(visited) < max_size:
            node, hop = queue.popleft()
            if hop >= max_hop:
                continue
            for neighbor in adj.get(node, []):
                if neighbor not in visited and len(visited) < max_size:
                    visited.add(neighbor)
                    queue.append((neighbor, hop + 1))
        return visited

    def _compute_cluster_emb(self, member_indices):
        if not member_indices:
            d = self.global_memory_embs[0].shape[0] if self.global_memory_embs else self.embedding_size
            return torch.zeros(d, device=self.device)
        embs = [self.global_memory_embs[m] for m in member_indices]
        cluster_emb = torch.stack(embs, dim=0).sum(dim=0)
        return cluster_emb / (cluster_emb.norm(p=2) + 1e-12)

    def update_link(self, update_info_list):
        """Update CF links: build user association graph based on memory similarity and cluster."""
        t0 = time.time()
        if not update_info_list or len(self.global_memory_embs) == 0:
            return
        adj = {}
        def add_edge(u, v):
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set()).add(u)
        base_nodes = set()
        for global_idx, cf_links in update_info_list:
            if not (0 <= global_idx < len(self.global_memory_embs)):
                continue
            base_nodes.add(global_idx)
            for link_idx in cf_links:
                if 0 <= link_idx < len(self.global_memory_embs):
                    add_edge(global_idx, link_idx)
        if not base_nodes:
            return
        all_nodes = set(adj.keys())
        for node in list(all_nodes):
            uid, mid = self.global_memory_info[node]
            uagent = self.user_agents.get(uid)
            if uagent and isinstance(uagent.link, dict) and mid in uagent.link:
                for cid in uagent.link[mid]:
                    if cid in self.clustered_memory:
                        for other in self.clustered_memory[cid]['link']:
                            if other != node and other in all_nodes:
                                add_edge(node, other)
        if not adj:
            print(f"update_link: no edges. time={time.time()-t0:.3f}s")
            return
        components, processed = [], set()
        for start in base_nodes:
            if start in processed:
                continue
            comp = self._bfs_limited_component(start, adj, self.max_hop)
            if len(comp) > 1:
                components.append(frozenset(comp))
                processed.update(comp)
        for comp in components:
            if len(comp) <= 1:
                continue
            comp_set = set(comp)
            existing_cid = None
            for cid, cdata in self.clustered_memory.items():
                if cdata.get('link') == comp_set:
                    existing_cid = cid
                    break
            cluster_emb = self._compute_cluster_emb(comp_set)
            if existing_cid is None:
                cid = len(self.clustered_memory)
                self.clustered_memory[cid] = {'emb': cluster_emb, 'link': comp_set}
            else:
                cid = existing_cid
                self.clustered_memory[cid]['emb'] = cluster_emb
                self.clustered_memory[cid]['link'] = comp_set
            for memidx in comp_set:
                uid, mid = self.global_memory_info[memidx]
                uagent = self.user_agents.get(uid)
                if uagent is None:
                    continue
                if not isinstance(uagent.link, dict):
                    uagent.link = {}
                if mid not in uagent.link:
                    uagent.link[mid] = [cid]
                elif cid not in uagent.link[mid]:
                    uagent.link[mid].append(cid)
        print(f"update_link: {len(base_nodes)} nodes, {len(components)} components. time={time.time()-t0:.3f}s")

    def get_linked_clusters(self, linked_list, item_emb_list):
        return linked_list

    def forward(self, batch_user, batch_pos_item, batch_neg_item, batch_id):
        """Forward pass: retrieve user memory and CF info, call LLM for preference prediction."""
        t0 = time.time()
        batch_size = batch_user.size(0)
        user_descriptions, pos_item_descriptions, neg_item_descriptions = [], [], []
        retrieved_mem_idx = []
        linked_memory, linked_inters = [], []
        for i, user in enumerate(batch_user):
            user_agent = self.user_agents[int(user)]
            pos_item_descriptions.append(self.item_descriptions[str(int(batch_pos_item[i]))])
            neg_item_descriptions.append(self.item_descriptions[str(int(batch_neg_item[i]))])
            if batch_id == 0:
                user_descriptions.append(user_agent.update_memory[-1])
            else:
                pos_idx = self.item_id_to_index[str(int(batch_pos_item[i]))]
                neg_idx = self.item_id_to_index[str(int(batch_neg_item[i]))]
                pos_emb, neg_emb = self.item_embeddings[pos_idx], self.item_embeddings[neg_idx]
                pos_mem_idx, _ = self.forward_retrieval(pos_emb, user_agent.memory_emb, k=self.forward_retrieval_k)
                neg_mem_idx, _ = self.forward_retrieval(neg_emb, user_agent.memory_emb, k=self.forward_retrieval_k)
                one_mem_idx = list(set(pos_mem_idx + neg_mem_idx))
                retrieved_mem_idx.append(one_mem_idx)
                user_descriptions.append([user_agent.memory_1[ii] for ii in one_mem_idx])
                one_linked_memory, one_linked_inters = [], []
                seen_linked = set()
                for memidx in one_mem_idx:
                    if isinstance(user_agent.link, dict) and memidx in user_agent.link:
                        all_clusters = self.get_linked_clusters(user_agent.link[memidx], [pos_emb, neg_emb])
                        for cluster_id in all_clusters:
                            if cluster_id not in self.clustered_memory:
                                continue
                            linked_atom_mem = list(self.clustered_memory[cluster_id]['link'])
                            linked_atom_mem_emb = [self.global_memory_embs[idx] for idx in linked_atom_mem]
                            for local_mem_idx in one_mem_idx:
                                global_mem_idx = self.get_global_idx(int(user), local_mem_idx)
                                if global_mem_idx is None:
                                    continue
                                refine_idx, _ = self.forward_retrieval(self.global_memory_embs[global_mem_idx], linked_atom_mem_emb, k=self.forward_retrieval_k)
                                for linked_atom_iter in refine_idx:
                                    actual_global_idx = linked_atom_mem[linked_atom_iter]
                                    if actual_global_idx in seen_linked:
                                        continue
                                    uid, mid = self.global_memory_info[actual_global_idx]
                                    if uid != int(user):
                                        seen_linked.add(actual_global_idx)
                                        one_linked_memory.append(self.user_agents[uid].memory_1[mid])
                                        one_linked_inters.append(self.user_agents[uid].intered_items[mid])
                linked_memory.append(one_linked_memory)
                linked_inters.append(one_linked_inters)
        if batch_id == 0:
            prompts = [self.rec_agent.astep_forward(int(batch_user[i]), user_descriptions[i], pos_item_descriptions[i], neg_item_descriptions[i]) for i in range(batch_size)]
        else:
            prompts = [self.rec_agent.astep_forward_linked(int(batch_user[i]), user_descriptions[i], pos_item_descriptions[i], neg_item_descriptions[i], linked_memory[i], linked_inters[i]) for i in range(batch_size)]
        responses = []
        for i in range(0, batch_size, self.api_batch):
            responses += asyncio.run(self.rec_agent.llm.agenerate_response(prompts[i:i + self.api_batch]))
        responses = [self.rec_agent.output_parser.parse(r['choices'][0]['message']['content']) for r in responses]
        print(f"forward: {time.time()-t0:.3f}s")
        return [r[0] for r in responses], [r[1] for r in responses], retrieved_mem_idx

    def backward(self, system_reasons, batch_user, batch_pos_item, batch_neg_item, retrieved_mem_idx):
        """Backward pass (wrong prediction): update user memory based on system feedback."""
        t0 = time.time()
        batch_size = len(batch_user)
        pos_item_titles = [self.item_text[int(batch_pos_item[i])] for i in range(batch_size)]
        neg_item_titles = [self.item_text[int(batch_neg_item[i])] for i in range(batch_size)]
        pos_descs = [self.item_descriptions[str(int(batch_pos_item[i]))] for i in range(batch_size)]
        neg_descs = [self.item_descriptions[str(int(batch_neg_item[i]))] for i in range(batch_size)]
        update_info_list = []
        if len(retrieved_mem_idx) == 0:
            prompts = [self.user_agents[int(batch_user[i])].astep_backward(system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_descs[i], neg_descs[i]) for i in range(batch_size)]
            responses = []
            for i in range(0, batch_size, self.chat_api_batch):
                responses += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(prompts[i:i + self.chat_api_batch]))
            descriptions = [self.user_agents[0].output_parser.parse_update(r["choices"][0]["message"]["content"]) for r in responses]
            for i, u in enumerate(batch_user):
                u = int(u)
                desc = descriptions[i][0] if isinstance(descriptions[i], list) else descriptions[i]
                self.user_agents[u].memory_1.append(desc)
                emb = self.generate_embedding(desc)
                self.user_agents[u].memory_emb.append(emb)
                self.user_agents[u].intered_items.append([pos_item_titles[i]])
                self.global_memory_embs.append(emb)
                new_mem_idx = len(self.user_agents[u].memory_1) - 1
                self.global_memory_info.append((u, new_mem_idx))
                global_idx = len(self.global_memory_info) - 1
                cf_links = self.find_cf_links(emb, u)
                update_info_list.append((global_idx, cf_links))
        else:
            user_mems = [[self.user_agents[int(batch_user[i])].memory_1[j] for j in retrieved_mem_idx[i]] for i in range(batch_size)]
            prompts = [self.user_agents[int(batch_user[i])].astep_backward_update(user_mems[i], system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_descs[i], neg_descs[i]) for i in range(batch_size)]
            responses = []
            for i in range(0, batch_size, self.chat_api_batch):
                responses += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(prompts[i:i + self.chat_api_batch]))
            parsed = [self.user_agents[0].output_parser.parse_judge_update(r["choices"][0]["message"]["content"]) for r in responses]
            judges = [[p[0]] for p in parsed]
            descs = [[p[1]] for p in parsed]
            for i, user in enumerate(batch_user):
                user = int(user)
                for j, judge in enumerate(judges[i]):
                    local_mem_idx = retrieved_mem_idx[i][j]
                    if judge is None:
                        self.user_agents[user].memory_1.append("mixup text")
                        emb = self.generate_embedding("mixup text")
                        self.user_agents[user].memory_emb.append(emb)
                        self.user_agents[user].intered_items.append([pos_item_titles[i]])
                        self.global_memory_embs.append(emb)
                        self.global_memory_info.append((user, len(self.user_agents[user].memory_1) - 1))
                        global_idx = len(self.global_memory_info) - 1
                        cf_links = self.find_cf_links(emb, user)
                        update_info_list.append((global_idx, cf_links))
                        print("backward judge is none " + pos_item_titles[i])
                    elif "irrelevant" in str(judge):
                        desc = descs[i][j]
                        self.user_agents[user].memory_1.append(desc)
                        emb = self.generate_embedding(desc)
                        self.user_agents[user].memory_emb.append(emb)
                        self.user_agents[user].intered_items.append([pos_item_titles[i]])
                        self.global_memory_embs.append(emb)
                        self.global_memory_info.append((user, len(self.user_agents[user].memory_1) - 1))
                        global_idx = len(self.global_memory_info) - 1
                        cf_links = self.find_cf_links(emb, user)
                        update_info_list.append((global_idx, cf_links))
                    elif 'incomplete' in str(judge):
                        desc = descs[i][j]
                        self.user_agents[user].memory_1[local_mem_idx] = desc
                        self.user_agents[user].intered_items[local_mem_idx].append(pos_item_titles[i])
                        emb = self.generate_embedding(desc)
                        self.user_agents[user].memory_emb[local_mem_idx] = emb
                        self.replace_memory_emb(user, local_mem_idx, emb)
                        global_idx = self.get_global_idx(user, local_mem_idx)
                        if global_idx is not None:
                            cf_links = self.find_cf_links(emb, user)
                            update_info_list.append((global_idx, cf_links))
        print(f"backward: {time.time()-t0:.3f}s")
        return update_info_list

    def backward_true(self, system_reasons, batch_user, batch_pos_item, batch_neg_item, retrieved_mem_idx):
        """Backward pass (correct prediction): consolidate and reinforce user memory."""
        t0 = time.time()
        batch_size = len(batch_user)
        pos_item_titles = [self.item_text[int(batch_pos_item[i])] for i in range(batch_size)]
        neg_item_titles = [self.item_text[int(batch_neg_item[i])] for i in range(batch_size)]
        pos_descs = [self.item_descriptions[str(int(batch_pos_item[i]))] for i in range(batch_size)]
        neg_descs = [self.item_descriptions[str(int(batch_neg_item[i]))] for i in range(batch_size)]
        update_info_list = []
        if len(retrieved_mem_idx) == 0:
            prompts = [self.user_agents[int(batch_user[i])].astep_backward_true(system_reasons[i], pos_item_titles[i], neg_item_titles[i], pos_descs[i], neg_descs[i]) for i in range(batch_size)]
        else:
            user_mems = [[self.user_agents[int(batch_user[i])].memory_1[j] for j in retrieved_mem_idx[i]] for i in range(batch_size)]
            prompts = [self.user_agents[int(batch_user[i])].astep_backward_consolid(system_reasons[i], user_mems[i], pos_item_titles[i], neg_item_titles[i], pos_descs[i], neg_descs[i]) for i in range(batch_size)]
        responses = []
        for i in range(0, batch_size, self.chat_api_batch):
            responses += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(prompts[i:i + self.chat_api_batch]))
        descriptions = [self.user_agents[0].output_parser.parse_update(r["choices"][0]["message"]["content"]) for r in responses]
        if len(retrieved_mem_idx) == 0:
            for i, user in enumerate(batch_user):
                user = int(user)
                desc = descriptions[i][0] if isinstance(descriptions[i], list) else descriptions[i]
                emb = self.generate_embedding(desc)
                if len(self.global_memory_embs) > 0:
                    global_link_id, _ = self.forward_retrieval(emb, self.global_memory_embs, k=1)
                    if global_link_id:
                        link_uid, link_mid = self.global_memory_info[global_link_id[0]]
                        if user == link_uid:
                            self.user_agents[user].memory_1[link_mid] = self.user_agents[user].memory_1[link_mid] + desc
                            emb_add = self.generate_embedding(self.user_agents[user].memory_1[link_mid])
                            self.user_agents[user].memory_emb[link_mid] = emb_add
                            self.replace_memory_emb(user, link_mid, emb_add)
                            self.user_agents[user].intered_items[link_mid].append(pos_item_titles[i])
                            cf_links = self.find_cf_links(emb_add, user)
                            update_info_list.append((global_link_id[0], cf_links))
                            continue
                self.user_agents[user].memory_1.append(desc)
                self.user_agents[user].memory_emb.append(emb)
                self.user_agents[user].intered_items.append([pos_item_titles[i]])
                self.global_memory_embs.append(emb)
                self.global_memory_info.append((user, len(self.user_agents[user].memory_1) - 1))
                global_idx = len(self.global_memory_info) - 1
                cf_links = self.find_cf_links(emb, user)
                update_info_list.append((global_idx, cf_links))
        else:
            for i, user in enumerate(batch_user):
                user = int(user)
                desc = descriptions[i][0] if isinstance(descriptions[i], list) else descriptions[i]
                local_mem_idx = retrieved_mem_idx[i][0]
                self.user_agents[user].memory_1[local_mem_idx] = desc
                self.user_agents[user].intered_items[local_mem_idx].append(pos_item_titles[i])
                self.user_agents[user].intered_items[local_mem_idx] = list(set(self.user_agents[user].intered_items[local_mem_idx]))
                emb = self.generate_embedding(desc)
                self.user_agents[user].memory_emb[local_mem_idx] = emb
                self.replace_memory_emb(user, local_mem_idx, emb)
                global_idx = self.get_global_idx(user, local_mem_idx)
                if global_idx is not None:
                    cf_links = self.find_cf_links(emb, user)
                    update_info_list.append((global_idx, cf_links))
        print(f"backward_true: {time.time()-t0:.3f}s")
        return update_info_list

    def convert_system_selections_to_accuracy(self, system_selections, pos_items, neg_items):
        accuracy = []
        for i, selection in enumerate(system_selections):
            pos_title = self.item_text[int(pos_items[i])]
            neg_title = self.item_text[int(neg_items[i])]
            matched, _ = process.extractOne(selection, [pos_title, neg_title])
            accuracy.append(1 if matched == pos_title else 0)
        return accuracy

    def calculate_loss(self, interaction):
        """Compute loss: iterate forward-backward until accuracy threshold is reached."""
        batch_user = interaction[self.USER_ID]
        batch_pos_item = interaction[self.ITEM_ID]
        batch_neg_item = interaction[self.NEG_ITEM_ID]
        for i in range(3):
            print(f"\n--------------{i}-th round--------------")
            system_selections, system_reasons, retrieved_mem_id = self.forward(batch_user, batch_pos_item, batch_neg_item, batch_id=i)
            accuracy = self.convert_system_selections_to_accuracy(system_selections, batch_pos_item, batch_neg_item)
            print(f"Accuracy: {sum(accuracy) / len(accuracy)} | ({sum(accuracy)}/{len(accuracy)})")
            if sum(accuracy) / len(accuracy) >= 0.95:
                break
            wrong_idx = [j for j, acc in enumerate(accuracy) if acc == 0]
            correct_idx = [j for j, acc in enumerate(accuracy) if acc == 1]
            all_update_info = []
            if wrong_idx:
                wrong_users = [int(batch_user[j]) for j in wrong_idx]
                print(f"Wrong Users: {wrong_users}")
                wrong_mem_idx = [retrieved_mem_id[j] for j in wrong_idx] if retrieved_mem_id else []
                update_info = self.backward([system_reasons[j] for j in wrong_idx], wrong_users, [int(batch_pos_item[j]) for j in wrong_idx], [int(batch_neg_item[j]) for j in wrong_idx], wrong_mem_idx)
                all_update_info.extend(update_info)
            if correct_idx:
                correct_users = [int(batch_user[j]) for j in correct_idx]
                print(f"Correct Users: {correct_users}")
                correct_mem_idx = [retrieved_mem_id[j] for j in correct_idx] if retrieved_mem_id else []
                update_info = self.backward_true([system_reasons[j] for j in correct_idx], correct_users, [int(batch_pos_item[j]) for j in correct_idx], [int(batch_neg_item[j]) for j in correct_idx], correct_mem_idx)
                all_update_info.extend(update_info)
            if all_update_info:
                self.update_link(all_update_info)

    def pairwise_train(self, interaction, batch_id):
        """Pairwise training: process pos/neg item pairs and update memory based on predictions."""
        batch_user = interaction[self.USER_ID]
        batch_pos_item = interaction[self.ITEM_ID]
        batch_neg_item = interaction[self.NEG_ITEM_ID]
        batch_size = batch_user.shape[0]
        for i in range(0, batch_size, self.api_batch):
            system_predictions, system_reasons, retrieved_mem_id = self.forward(batch_user[i:i + self.api_batch], batch_pos_item[i:i + self.api_batch], batch_neg_item[i:i + self.api_batch], batch_id)
            correct_idx, wrong_idx = [], []
            for j, pred in enumerate(system_predictions):
                if pred == 'A':
                    correct_idx.append(j)
                else:
                    wrong_idx.append(j)
            print(f"Correct: {len(correct_idx)}, Wrong: {len(wrong_idx)}")
            all_update_info = []
            if wrong_idx:
                wrong_users = [int(batch_user[j]) for j in wrong_idx]
                print(f"Wrong Users: {wrong_users}")
                wrong_mem_idx = [retrieved_mem_id[j] for j in wrong_idx] if retrieved_mem_id else []
                update_info = self.backward([system_reasons[j] for j in wrong_idx], wrong_users, [int(batch_pos_item[j]) for j in wrong_idx], [int(batch_neg_item[j]) for j in wrong_idx], wrong_mem_idx)
                all_update_info.extend(update_info)
            if correct_idx:
                correct_users = [int(batch_user[j]) for j in correct_idx]
                print(f"Correct Users: {correct_users}")
                correct_mem_idx = [retrieved_mem_id[j] for j in correct_idx] if retrieved_mem_id else []
                update_info = self.backward_true([system_reasons[j] for j in correct_idx], correct_users, [int(batch_pos_item[j]) for j in correct_idx], [int(batch_neg_item[j]) for j in correct_idx], correct_mem_idx)
                all_update_info.extend(update_info)
            if all_update_info:
                self.update_link(all_update_info)

    def get_user_summary(self, user_id, batch_size):
        user_memory = [self.user_agents[int(u)].memory_1 for u in user_id]
        user_memory_numinter = [len(self.user_agents[int(u)].intered_items) for u in user_id]
        prompts = [self.user_agents[int(user_id[i])].astep_inference(user_memory[i], user_memory_numinter[i]) for i in range(len(user_id))]
        responses = []
        for i in range(0, batch_size, self.chat_api_batch):
            responses += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(prompts[i:i+self.chat_api_batch]))
        return [self.user_agents[0].output_parser.parse_summary_update(r["choices"][0]["message"]["content"]) for r in responses]

    def full_sort_predict(self, interaction, idxs):
        """Full sort prediction: rank all candidate items and return score matrix."""
        batch_size = idxs.shape[0]
        batch_user = interaction['user_id']
        if self.config['evaluation'] == 'rag' and self.config['item_representation'] == 'rag':
            batch_user_descriptions = [self.user_agents[int(batch_user[i])].memory_1[-1] for i in range(batch_size)]
            batch_user_embedding = self.generate_embedding(batch_user_descriptions)
        else:
            batch_user_embedding = None
        scores = torch.full((batch_user.shape[0], self.n_items), -10000.)
        user_memorys, linked_memorys, linked_inters = [], [], []
        list_of_item_descriptions, candidate_texts, user_his_texts = [], [], []
        for i in range(batch_size):
            user_his_text, candidate_text, candidate_text_order, candidate_idx, candidate_text_order_description, user_memory, linked_memory, linked_inter = self.get_batch_inputs(int(batch_user[i]), interaction, idxs, i, batch_user_embedding)
            user_memorys.append(user_memory)
            linked_memorys.append(linked_memory)
            linked_inters.append(linked_inter)
            user_his_texts.append(user_his_text)
            list_of_item_descriptions.append('\n\n'.join(candidate_text_order_description))
            candidate_texts.append(candidate_text)
        evaluation_prompts, messages = self.evaluation(batch_user, user_his_texts, list_of_item_descriptions, user_memorys, linked_memorys, linked_inters)
        self.parsing_output_text(scores, messages, idxs, candidate_texts, interaction[self.ITEM_ID])
        return scores

    def evaluation(self, batch_user, user_his_texts, list_of_item_descriptions, user_memorys, linked_memorys, linked_inters):
        """Call LLM to evaluate and rank candidate items."""
        batch_size = batch_user.shape[0]
        prompts = [self.rec_agent.astep_evaluation(int(batch_user[i]), user_his_texts[i], list_of_item_descriptions[i], user_memorys[i], linked_memorys[i], linked_inters[i]) for i in range(batch_size)]
        messages = []
        for i in tqdm(range(0, batch_size, self.chat_api_batch)):
            messages += asyncio.run(self.user_agents[0].llm_chat.agenerate_response_without_construction(prompts[i:i+self.chat_api_batch]))
        return prompts, [self.rec_agent.output_parser.parse_evaluation(r["choices"][0]["message"]["content"]) for r in messages]

    def get_batch_inputs(self, user_id, interaction, idxs, i, user_embedding):
        """Prepare evaluation inputs: user history, candidates, user memory and CF info."""
        user_his = interaction[self.ITEM_SEQ]
        user_his_len = interaction[self.ITEM_SEQ_LEN]
        real_his_len = min(self.max_his_len, user_his_len[i].item())
        user_his_text = [str(j+1) + '. ' + self.item_text[user_his[i, user_his_len[i].item() - real_his_len + j].item()] for j in range(real_his_len)]
        candidate_text = [self.item_text[idxs[i, j]] for j in range(idxs.shape[1])]
        user_agent = self.user_agents[user_id]
        user_mem_content_list = []
        linked_mem_content_list, linked_inter_list = [], []
        user_mem_idx_list = set()
        cluster_idx_list = []
        for candidate_item in idxs[i]:
            candidate_emb = self.item_embeddings[self.item_id_list.index(str(int(candidate_item)))]
            mem_idx, _ = self.forward_retrieval(candidate_emb, user_agent.memory_emb, k=self.forward_retrieval_k)
            if mem_idx:
                user_mem_idx_list.add(mem_idx[0])
                if isinstance(user_agent.link, dict) and mem_idx[0] in user_agent.link:
                    all_clusters = self.get_linked_clusters(user_agent.link[mem_idx[0]], [candidate_emb])
                    cluster_idx_list = list(set(cluster_idx_list + all_clusters))
        for mem_idx in user_mem_idx_list:
            user_mem_content_list.append(user_agent.memory_1[mem_idx])
        seen_linked = set()
        for cluster_id in cluster_idx_list:
            if cluster_id not in self.clustered_memory:
                continue
            cluster_link_idx = list(self.clustered_memory[cluster_id]['link'])
            for link in cluster_link_idx:
                if link in seen_linked:
                    continue
                seen_linked.add(link)
                uid, mid = self.global_memory_info[link]
                if uid != user_id:
                    linked_mem_content_list.append(self.user_agents[uid].memory_1[mid])
                    linked_inter_list.append(self.user_agents[uid].intered_items[mid])
        candidate_text_order = [str(j + 1) + '. ' + self.item_text[idxs[i, j].item()] for j in range(idxs.shape[1])]
        if self.config['item_representation'] == 'direct':
            candidate_text_order_description = [str(j + 1) + '. ' + self.item_text[idxs[i, j].item()] + ': ' + self.item_descriptions[str(idxs[i, j].item())] for j in range(idxs.shape[1])]
        elif self.config['item_representation'] == 'rag' and self.config['evaluation'] == 'rag':
            item_descriptions = []
            for item in idxs[i]:
                item = int(item)
                item_embeddings = list(self.item_agents[item].memory_embedding.values())
                distances = distances_from_embeddings(user_embedding, item_embeddings)
                indices_of_nearest_neighbors = indices_of_nearest_neighbors_from_distances(distances)[0]
                item_descriptions.append(list(self.item_agents[item].memory_embedding.keys())[indices_of_nearest_neighbors])
            candidate_text_order_description = [str(j+1) + '. ' + self.item_text[idxs[i, j].item()] + ': ' + item_descriptions[j] for j in range(idxs.shape[1])]
        else:
            candidate_text_order_description = candidate_text_order
        return user_his_text, candidate_text, candidate_text_order, idxs[i].tolist(), candidate_text_order_description, user_mem_content_list, linked_mem_content_list, linked_inter_list

    def parsing_output_text(self, scores, messages, idxs, candidate_texts, batch_pos_item):
        """Parse LLM ranking output, compute scores, and update diversity metrics."""
        for i, message in enumerate(messages):
            candidate_text = candidate_texts[i]
            ranked_item_ids = []
            for j, item_detail in enumerate(message):
                if len(item_detail) < 1 or item_detail.endswith('candidate movies:'):
                    continue
                pr = item_detail.find('. ')
                item_name = item_detail[pr + 2:].strip() if item_detail[:pr].isdigit() else item_detail.strip()
                if self.config['match_rule'] == 'exact':
                    for idx, ct in enumerate(candidate_text):
                        if ct in item_name:
                            item_id = idxs[i, idx]
                            if scores[i, item_id] > -5000.:
                                break
                            scores[i, item_id] = self.config['recall_budget'] - j
                            ranked_item_ids.append(int(item_id))
                            break
                elif self.config['match_rule'] == 'fuzzy':
                    matched, _ = process.extractOne(item_name, candidate_text)
                    matched_idx = candidate_text.index(matched)
                    item_id = idxs[i, matched_idx]
                    if scores[i, item_id] <= -5000.:
                        scores[i, item_id] = self.config['recall_budget'] - j
                        ranked_item_ids.append(int(item_id))
            if len(ranked_item_ids) >= 5:
                self.diversity_metrics['ild_list'].append(self.compute_ild(ranked_item_ids, top_k=5))
                self.diversity_metrics['entropy_list'].append(self.compute_entropy(ranked_item_ids, top_k=5))
                self.diversity_metrics['ds_list'].append(self.compute_diversity_score(ranked_item_ids, top_k=5))
