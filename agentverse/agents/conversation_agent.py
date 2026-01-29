#am
from __future__ import annotations
import logging
from logging import getLogger
import bdb
from string import Template
from typing import TYPE_CHECKING, List, Optional
import random
from agentverse.message import Message
from pydantic import BaseModel, Field
from . import agent_registry
from .base import BaseAgent
import time
from collections import defaultdict
from agentverse.llms import BaseLLM


logger = getLogger()
@agent_registry.register("embeddingagent")
class EmbeddingAgent(BaseAgent):
    def step(self, env_description: str = "") -> Message:
        """Get one step response"""
        pass

    def astep(self, env_description: str = "") -> Message:
        """Asynchronous version of step"""
        pass

    def add_message_to_memory(self, messages: List[Message]) -> None:
        """Add a message to the memory"""
        pass

    def reset(self) -> None:
        """Reset the agent"""
        pass
    async def astep_forward(self, sentence: str = "") -> Message:
        """Asynchronous version of step in forward process"""
        sentence = '{}'.format(sentence)
        for i in range(self.max_retry):
            try:
            # print(prompt)
            # print("!!!!!!!!!")
                response = await self.llm.agenerate_response(sentence)
            # print(parsed_response)
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                logger.info(e)
                logger.info("Retrying...")
                time.sleep(30)
                continue

        if response is None:
            logger.info(f"{self.name} failed to generate valid response.")

        return response



@agent_registry.register("recagent")
class RecAgent(BaseAgent):
    llm_chat: BaseLLM
    system_prompt_template_backward: str
    system_prompt_template_evaluation_basic: str
    system_prompt_template_evaluation_retrieval: str
    system_prompt_template_evaluation_sequential: str
    prompt_template_clustered : str
    prompt_template_linked: str
    n_users: int = 500
    memory: list = []
    user_examples = defaultdict(dict)

    user_id2memory: dict = defaultdict(list)
    def step(self, env_description: str = "") -> Message:
        prompt = self._fill_prompt_template(env_description)

        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = self.llm.generate_response(prompt)
                parsed_response = self.output_parser.parse(response)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.info(e)
                logger.info("Retrying...")
                time.sleep(30)
                continue

        if parsed_response is None:
            logger.info(f"{self.name} failed to generate valid response.")

        message = Message(
            content=""
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message

    async def astep(self, env_description: str = "") -> Message:
        """Asynchronous version of step"""
        prompt = self._fill_prompt_template(env_description)

        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(prompt)
                parsed_response = self.output_parser.parse(response)
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                logger.info(e)
                logger.info("Retrying...")
                time.sleep(30)
                continue

        if parsed_response is None:
            logger.info(f"{self.name} failed to generate valid response.")

        message = Message(
            content=""
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message

    def astep_forward(self, user_id: int, user_description: str = "", pos_item_description: str = "", neg_item_description: str = "") -> Message:
        """Asynchronous version of step in forward process"""
        prompt = self._fill_prompt_template(user_id, user_description, pos_item_description, neg_item_description)
        return prompt
    
    #int(batch_user[i]), user_descriptions[i], pos_item_descriptions[i], neg_item_descriptions[i], linked_memory[i], linked_inters[i], clustered_summary[i]                                                                          
    def astep_forward_linked(self, user_id: int, user_description: str = "", pos_item_description: str = "", neg_item_description: str = ""
                             , linked_memory: str="", linked_inters: str="") -> Message:
        prompt = self._fill_prompt_template_linked(user_id, user_description, pos_item_description, neg_item_description
                                                   , linked_memory, linked_inters)
        return prompt
    
                            
    def astep_evaluation(self, user_id, user_his_text, list_of_item_description, user_memorys, linked_memorys, linked_inters):
        """Asynchronous version of step in forward process"""
        prompt = self._fill_prompt_template_evaluation(user_id, user_his_text, list_of_item_description, user_memorys, linked_memorys, linked_inters)
        return prompt


    def astep_backward(self, user_id: int,  truth_or_falsity: int, message: tuple = (), user_his_text: str = "",pos_item_titles: str = "", pos_item_description: str = "", neg_item_description: str = "", user_description: str = "",user_selection_reasons: str = "") -> Message:
        """Asynchronous version of step in forward process"""
        prompt = self._fill_prompt_template_backward(user_id,  truth_or_falsity, message, user_his_text, pos_item_titles, pos_item_description, neg_item_description, user_description, user_selection_reasons)
        return prompt
    
    def summary_clustered(self, clustered_node, content):
        """Asynchronous version of step in forward process"""
        prompt = self._fill_prompt_template_clustered(clustered_node, content)
        return prompt
 

    def _fill_prompt_template(self, user_id, user_description: str = "", pos_item_description: str = "", neg_item_description: str = "") -> str:
        # forward prompt
        input_arguments = {
            "list_of_item_description":  f'1. {neg_item_description} \n 2. {pos_item_description}',
            "user_memory": user_description,
        }
        return Template(self.prompt_template).safe_substitute(input_arguments)
    
    # 总结clustered节点的summary
    def _fill_prompt_template_clustered(self, clustered_node, content):
        bottom_memory = ', '.join([f"{i+1}. {memory}" for i, memory in enumerate(content)])
        input_arguments = {
            
            "bottom_memory": bottom_memory,
        }
        return Template(self.prompt_template_clustered).safe_substitute(input_arguments)
    
    # linked 推荐提示
    def _fill_prompt_template_linked(self, user_id, user_description: str = "", pos_item_description: str = "", neg_item_description: str = "",
                                     linked_memory: str="", linked_inters: str="") -> str:
        linked_info = ""
        for memory_idx in range(len(linked_memory)):
            memory = str(linked_memory[memory_idx])
            interaction = str(linked_inters[memory_idx])
            linked_info += f"The related memory is '{memory}', and the corresponding CD interaction is {interaction}."
        input_arguments = {
            "list_of_item_description":  f'1. {neg_item_description} \n 2. {pos_item_description}',
            "user_memory": user_description,
            
            "linked_info": linked_info, 
        }
        return Template(self.prompt_template_linked).safe_substitute(input_arguments)

    # user_id, user_his_text, list_of_item_description, user_memorys, linked_memorys, linked_inters
    def _fill_prompt_template_evaluation(self, user_id, user_his_text, list_of_item_description,
                                         user_memory, linked_memory, linked_inters) -> str:
        linked_info = ""
        for memory_idx in range(len(linked_memory)):
            memory = str(linked_memory[memory_idx])
            interaction = str(linked_inters[memory_idx])
            linked_info += f"The related memory is '{memory}', and the corresponding CD interaction is {interaction}."

        system_role = {"role": "system",
                        "content": "You are a recommender system. During interactions, a user will provide a self-introduction that includes their preferences and dislikes, along with a list of candidate CDs. Your task is to reorder this list of CDs, prioritizing those that align with the user's preferences at the top and placing those that the user dislikes at the bottom."}
        input_arguments = {
            # "user_description": user_description, # 生成的总结
            "historical_interactions": '\n'.join(user_his_text),
            "candidate_num": 10,
            "example_list_of_item_description": list_of_item_description,
            "user_memory": user_memory, # 相关的记忆
            "linked_info": linked_info,
        }
        # if user_his_text != []:
        #     return [system_role, {"role": "user", "content": Template(self.system_prompt_template_evaluation_sequential).safe_substitute(input_arguments)}]
        # else:
        return [system_role, {"role": "user", "content": Template(self.system_prompt_template_evaluation_basic).safe_substitute(input_arguments)}]

    def _fill_prompt_template_backward(self, user_id: int, truth_or_falsity: int, message: tuple = (), user_his_text: str = "", pos_item_titles: str = "", pos_item_description: str = "", neg_item_description: str = "", user_description: str = "",user_selection_reasons: str = "") -> str:


        if truth_or_falsity:
            recommended_truth_or_falsity = 'suitable'
        else:
            recommended_truth_or_falsity = 'unsuitable'

        if random.random() > 0.5:
            input_arguments = {
                "user_description": user_description,
                # "user_his_text": user_his_text,
                "item_description_1": pos_item_description,
                "item_description_2": neg_item_description,
                "recommendation_strategy": self.user_id2memory[user_id][-1] if len(self.user_id2memory[user_id]) else "Analyze the user's preference and aversions based on his self-description. Compare the difference between the two candidate CDs. Select the CD that meets the user's preference and goes against the user's aversions.",
                "recommended_movie": message[0],

                # "recommended_rationale": message[1],
                "pos_movie": pos_item_titles,
                "user_reasons": user_selection_reasons,
                "truth_or_falsity": recommended_truth_or_falsity,
                "truth_or_falsity": recommended_truth_or_falsity,
                # "user_feedback": feedback,
            }
        else:
            input_arguments = {
                "user_description": user_description,
                # "user_his_text": user_his_text,
                "item_description_1": neg_item_description,
                "item_description_2": pos_item_description,
                "recommendation_strategy": self.user_id2memory[user_id][-1] if len(self.user_id2memory[user_id]) else "Analyze the user's preference and aversions based on his self-description. Compare the difference between the two candidate CDs. Select the CD that meets the user's preference and goes against the user's aversions.",
                "recommended_movie": message[0],
                # "recommended_rationale": message[1],
                "pos_movie": pos_item_titles,
                 "user_reasons": user_selection_reasons,
                "truth_or_falsity": recommended_truth_or_falsity,
                "truth_or_falsity": recommended_truth_or_falsity,
                # "user_feedback": feedback,
            }
        return Template(self.system_prompt_template_backward).safe_substitute(input_arguments)

    def add_message_to_memory(self, messages: List[str]) -> None:
        self.memory.extend(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver


@agent_registry.register("useragent")
class UserAgent(BaseAgent):
    llm_chat: BaseLLM
    role_description: dict
    role_description_string_1: str
    user_prompt_system_role: str
    user_prompt_template_true: str
    user_prompt_template_consolid: str
    user_prompt_template_update: str
    user_prompt_template_inference: str
    role_description_string_3: str
    memory_1: list = []
    memory_emb: list = []
    intered_items: list = []
    link: dict = {}
    update_memory: list = []
    feedback: list = []
    historical_interactions: dict = {}
    

    naive_memory: str = ""  
    interacted_items: list = [] 
    

    def step(self, env_description: str = "") -> Message:
        prompt = self._fill_prompt_template(env_description)

        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = self.llm.generate_response(prompt)
                parsed_response = self.output_parser.parse(response)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logger.info(e)
                logger.info("Retrying...")
                time.sleep(30)
                continue

        if parsed_response is None:
            logger.info(f"{self.name} failed to generate valid response.")

        message = Message(
            content=""
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message

    async def astep(self, env_description: str = "") -> Message:
        """Asynchronous version of step"""
        prompt = self._fill_prompt_template(env_description)

        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(prompt)
                parsed_response = self.output_parser.parse(response)
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                logger.info(e)
                logger.info("Retrying...")
                time.sleep(30)
                continue

        if parsed_response is None:
            logger.info(f"{self.name} failed to generate valid response.")

        message = Message(
            content=""
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message
    
    
    def astep_backward(self, system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "") -> Message:
        """Asynchronous version of step in forward process"""
        prompt = self._fill_prompt_template_backward(system_reasons, pos_item_title, neg_item_title, pos_item_description, neg_item_description)
        return prompt
    
    
    def astep_backward_true(self, system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "") -> Message:
        """Asynchronous version of step in forward process"""
        prompt = self._fill_prompt_template_backward_true(system_reasons, pos_item_title, neg_item_title, pos_item_description, neg_item_description)
        return prompt
    
   
    def astep_backward_update(self, user_memory, system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "") -> Message:
        """Asynchronous version of step in forward process"""
        prompt = self._fill_prompt_template_backward_update(user_memory, system_reasons, pos_item_title, neg_item_title, pos_item_description, neg_item_description)
        return prompt
    
    
    def astep_backward_consolid(self, system_reasons: str = "", user_memory: str="", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "") -> Message:
        """Asynchronous version of step in forward process"""
        prompt = self._fill_prompt_template_backward_consolid(system_reasons, user_memory, pos_item_title, neg_item_title, pos_item_description, neg_item_description)
        return prompt
    
    def astep_inference(self, user_memory, user_memory_numintered):
        """Asynchronous version of step in forward process"""
        prompt = self._fill_prompt_template_inference(user_memory, user_memory_numintered)
        return prompt

    def _fill_prompt_template_backward(self, system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "") -> str:
        system_role = {"role":"system","content":Template(self.user_prompt_system_role).safe_substitute({"user_description": self.update_memory[-1]})}
        input_arguments = {
            "list_of_item_description": f'1. {neg_item_description} \n 2. {pos_item_description}',
            "system_reason": system_reasons,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            "pos_item_title": pos_item_title,
        }
        user_role = {"role":"user", "content": Template(self.prompt_template).safe_substitute(input_arguments)}
        return [system_role, user_role]
    
    def _fill_prompt_template_backward_true(self, system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "") -> str:
        system_role = {"role":"system","content":Template(self.user_prompt_system_role).safe_substitute({"user_description": self.update_memory[-1]})}
        input_arguments = {
            "list_of_item_description": f'1. {neg_item_description} \n 2. {pos_item_description}',
            "system_reason": system_reasons,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            "pos_item_title": pos_item_title,
        }
        user_role = {"role":"user", "content": Template(self.user_prompt_template_true).safe_substitute(input_arguments)}
        return [system_role, user_role]
    
    def _fill_prompt_template_backward_update(self, user_memory, system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "") -> str:
        system_role = {"role":"system","content":Template(self.user_prompt_system_role).safe_substitute({"user_description": self.update_memory[-1]})}
        expand_user_memory = ', '.join([f"{i+1}. {memory}" for i, memory in enumerate(user_memory)])
        input_arguments = {
            "list_of_item_description": f'1. {neg_item_description} \n 2. {pos_item_description}',
            "user_memory": expand_user_memory,
            "pos_item_title": pos_item_title,
            "system_reason": system_reasons,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            "pos_item_title": pos_item_title,
        }
        user_role = {"role":"user", "content": Template(self.user_prompt_template_update).safe_substitute(input_arguments)}
        return [system_role, user_role]
    
    def _fill_prompt_template_backward_consolid(self, system_reason, user_memory, pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "") -> str:
        system_role = {"role":"system","content":Template(self.user_prompt_system_role).safe_substitute({"user_description": self.update_memory[-1]})}
        expand_user_memory = ', '.join([f"{i+1}. {memory}" for i, memory in enumerate(user_memory)])
        input_arguments = {
            "list_of_item_description": f'1. {neg_item_description} \n 2. {pos_item_description}',
            "user_memory": expand_user_memory,
            "pos_item_title": pos_item_title,
            "system_reason": system_reason,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            "pos_item_title": pos_item_title,
        }
        user_role = {"role":"user", "content": Template(self.user_prompt_template_consolid).safe_substitute(input_arguments)}
        return [system_role, user_role]
    
    def _fill_prompt_template_inference(self, user_memory, user_memory_numintered):
        system_role = {"role":"system","content":Template(self.user_prompt_system_role).safe_substitute({"user_description": self.update_memory[-1]})}
        
        user_memory_sorted = [
            m for m, _ in sorted(
                zip(user_memory, user_memory_numintered),
                key=lambda x: x[1],
                reverse=True
            )
        ]
        expand_hot_memory = ', '.join([f"{i+1}. {memory}" for i, memory in enumerate(user_memory_sorted[:len(user_memory)/2])])
        expand_cold_memory = ', '.join([f"{i+1}. {memory}" for i, memory in enumerate(user_memory_sorted[len(user_memory)/2:])])
        input_arguments = {
           "hot_mem":expand_hot_memory,
           "cold_mem":expand_cold_memory
        }
        user_role = {"role":"user", "content": Template(self.user_prompt_template_inference).safe_substitute(input_arguments)}
        return [system_role, user_role]


    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver


@agent_registry.register("itemagent")
class ItemAgent(BaseAgent):
    llm_chat: BaseLLM
    role_description: dict
    role_description_string: str
    item_prompt_template_true: str
    memory_embedding: dict
    memory: list = []
    update_memory: list = []
    memory_review: dict = {}

    def step(self, env_description: str = "") -> Message:
        prompt = self._fill_prompt_template(env_description)

        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = self.llm.generate_response(prompt)
                parsed_response = self.output_parser.parse(response)
                break
            except KeyboardInterrupt:
                raise
            except Exception as e:
                logging.info(e)
                logging.info("Retrying...")
                time.sleep(30)
                continue

        if parsed_response is None:
            logging.info(f"{self.name} failed to generate valid response.")

        message = Message(
            content=""
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message

    async def astep(self, env_description: str = "") -> Message:
        """Asynchronous version of step"""
        prompt = self._fill_prompt_template(env_description)

        parsed_response = None
        for i in range(self.max_retry):
            try:
                response = await self.llm.agenerate_response(prompt)
                parsed_response = self.output_parser.parse(response)
                break
            except (KeyboardInterrupt, bdb.BdbQuit):
                raise
            except Exception as e:
                logger.info(e)
                logger.info("Retrying...")
                time.sleep(30)
                continue

        if parsed_response is None:
            logger.info(f"{self.name} failed to generate valid response.")

        message = Message(
            content=""
            if parsed_response is None
            else parsed_response.return_values["output"],
            sender=self.name,
            receiver=self.get_receiver(),
        )
        return message

    def astep_backward(self,  system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "", user_description: str="") -> Message:
        """Asynchronous version of step in forward process"""

        prompt = self._fill_prompt_template_backward_item(system_reasons, pos_item_title, neg_item_title, pos_item_description, neg_item_description, user_description)
        return prompt

    def astep_backward_true(self,  system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "", user_description: str="") -> Message:
        """Asynchronous version of step in forward process"""

        prompt = self._fill_prompt_template_backward_item_true(system_reasons, pos_item_title, neg_item_title, pos_item_description, neg_item_description, user_description)
        return prompt

    def astep_update(self, similar_item_descriptions: list):
        prompt = self._fill_prompt_template_update( similar_item_descriptions)


    def _fill_prompt_template_update(self, similar_item_descriptions):
        input_arguments = {
            "item_role_description": self.role_description_string,
            "list_of_similar_item_description": '\n'.join([str(j + 1) + '. ' + similar_item_descriptions[j]
                                for j in range(len(similar_item_descriptions))])
        }
        return Template(self.prompt_template_update).safe_substitute(input_arguments)

    def _fill_prompt_template_backward_item(self, system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "", user_description: str="") -> str:
        """Fill the placeholders in the prompt template

        In the conversation agent, three placeholders are supported:
        - ${agent_name}: the name of the agent
        - ${env_description}: the description of the environment
        - ${role_description}: the description of the role of the agent
        - ${chat_history}: the chat history of the agent
        """

        input_arguments = {
            "user_description": user_description,
            "list_of_item_description": f'1. {neg_item_description} \n 2. {pos_item_description}',
            "neg_item_title": neg_item_title,
            "system_reason": system_reasons,
            "pos_item_title": pos_item_title,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            }
        return Template(self.prompt_template).safe_substitute(input_arguments)



    def _fill_prompt_template_backward_item_true(self, system_reasons: str = "", pos_item_title: str="", neg_item_title: str="", pos_item_description: str="", neg_item_description: str = "", user_description: str="") -> str:
        """Fill the placeholders in the prompt template

        In the conversation agent, three placeholders are supported:
        - ${agent_name}: the name of the agent
        - ${env_description}: the description of the environment
        - ${role_description}: the description of the role of the agent
        - ${chat_history}: the chat history of the agent
        """

        input_arguments = {
            "user_description": user_description,
            "list_of_item_description": f'1. {neg_item_description} \n 2. {pos_item_description}',
            "pos_item_title": pos_item_title,
            "pos_item_title": pos_item_title,
            "neg_item_title": neg_item_title,
            }
        return Template(self.item_prompt_template_true).safe_substitute(input_arguments)



    def add_message_to_memory(self, messages: List[Message]) -> None:
        self.memory.add_message(messages)

    def reset(self) -> None:
        """Reset the agent"""
        self.memory.reset()
        # TODO: reset receiver